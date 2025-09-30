import sys
import os
import sqlite3
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torch import Generator
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QRadioButton, QButtonGroup,
                             QFileDialog, QMessageBox, QGroupBox, QGridLayout, QDialog,
                             QSpinBox, QDialogButtonBox, QFormLayout, QProgressBar,
                             QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
                             QCheckBox, QComboBox, QLineEdit, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QRect, QPoint, QSize
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush, QColor, QIntValidator, QPainter, QPen, QMouseEvent
from typing import List, Tuple, Optional, Union, Dict, Any, Set
import re
from pathlib import Path
import json
import base64
from io import BytesIO
import pickle

from matplotlib.colors import LinearSegmentedColormap

def colormap():
    """Define a custom colormap for HiC visualization"""
    return LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelingDatabaseSchema:
    """SQLite database schema for the labeling database (separate from source data)"""
    
    CREATE_TABLES = [
        """
        CREATE TABLE IF NOT EXISTS source_databases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL UNIQUE,
            description TEXT,
            total_samples INTEGER DEFAULT 0,
            labeled_samples INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS sample_indices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_db_id INTEGER NOT NULL,
            key_id INTEGER NOT NULL,
            original_index INTEGER NOT NULL,
            split_type TEXT,  -- 'train', 'val', 'test'
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (source_db_id) REFERENCES source_databases (id),
            UNIQUE(source_db_id, key_id)
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_db_id INTEGER NOT NULL,
            key_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            labeled_by TEXT,
            labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            FOREIGN KEY (source_db_id) REFERENCES source_databases (id),
            UNIQUE(source_db_id, key_id)
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS label_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            color TEXT,  -- Hex color code
            sort_order INTEGER DEFAULT 0
        )
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_sample_indices_source ON sample_indices(source_db_id);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_sample_indices_key ON sample_indices(key_id);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_labels_source_key ON labels(source_db_id, key_id);
        """
    ]
    
    DEFAULT_CATEGORIES = [
        ("Strong Positive", "High confidence positive signal", "#2E8B57", 1),
        ("Weak Positive", "Low confidence positive signal", "#90EE90", 2),
        ("Weak Negative", "Low confidence negative signal", "#FFA500", 3),
        ("Negative", "Clear negative signal", "#FF6347", 4),
        ("Noise", "Noisy or unclear signal", "#808080", 5)
    ]


class SourceDatabaseInspector:
    """Inspects SQLite3 source databases to understand their structure"""
    
    @staticmethod
    def inspect_database(db_path: str) -> Dict[str, Any]:
       """Inspect a SQLite3 database and return structure information"""
       try:
           conn = sqlite3.connect(db_path)
           cursor = conn.cursor()
           
           # Get all tables
           cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
           tables = [row[0] for row in cursor.fetchall()]
           
           info = {
               'file_path': db_path,
               'tables': tables,
               'has_target_table': 'imag_with_seqs' in tables,
               'table_columns': [],
               'table_count': 0,
               'key_id_range': (None, None),
               'sample_key_ids': [],
               'has_labels_column': False,
               'needs_labels_column': False,
               'available_resolutions': [],
               'available_datasets': []
           }
           
           # If imag_with_seqs table exists, get its structure
           if info['has_target_table']:
               # Get column information
               cursor.execute("PRAGMA table_info(imag_with_seqs)")
               columns = cursor.fetchall()
               info['table_columns'] = [col[1] for col in columns]  # col[1] is column name
               
               # Check if labels column exists
               info['has_labels_column'] = 'labels' in info['table_columns']
               info['needs_labels_column'] = not info['has_labels_column']
               
               # Check for and create indices if missing for performance
               cursor.execute("""
                   SELECT name FROM sqlite_master 
                   WHERE type='index' AND tbl_name='imag_with_seqs'
               """)
               existing_indices = {row[0].lower() for row in cursor.fetchall()}
               
               indices_created = []
               
               # Create composite index for smart filtering if missing
               if not any('dataset' in idx and 'hic_path' in idx and 'resolution' in idx 
                         for idx in existing_indices):
                   try:
                       cursor.execute("""
                           CREATE INDEX IF NOT EXISTS idx_dataset_hic_res 
                           ON imag_with_seqs(dataset, hic_path, resolution, key_id)
                       """)
                       conn.commit()
                       indices_created.append("idx_dataset_hic_res")
                   except:
                       pass
               
               # Create key_id index if missing
               if not any('key_id' in idx for idx in existing_indices):
                   try:
                       cursor.execute("""
                           CREATE INDEX IF NOT EXISTS idx_key_id 
                           ON imag_with_seqs(key_id)
                       """)
                       conn.commit()
                       indices_created.append("idx_key_id")
                   except:
                       pass
               
               # Create labels index if column exists
               if info['has_labels_column'] and not any('labels' in idx for idx in existing_indices):
                   try:
                       cursor.execute("""
                           CREATE INDEX IF NOT EXISTS idx_labels 
                           ON imag_with_seqs(labels)
                       """)
                       conn.commit()
                       indices_created.append("idx_labels")
                   except:
                       pass
               
               if indices_created:
                   print(f"Created indices for faster queries: {', '.join(indices_created)}")
                   info['indices_created'] = indices_created
               
               # Get count
               cursor.execute("SELECT COUNT(*) FROM imag_with_seqs")
               info['table_count'] = cursor.fetchone()[0]
               
               # Get key_id range if key_id column exists
               if 'key_id' in info['table_columns']:
                   cursor.execute("SELECT MIN(key_id), MAX(key_id) FROM imag_with_seqs")
                   min_key, max_key = cursor.fetchone()
                   info['key_id_range'] = (min_key, max_key)
                   
                   # Get sample of key_ids (first 1000)
                   cursor.execute("SELECT key_id FROM imag_with_seqs ORDER BY key_id LIMIT 1000")
                   info['sample_key_ids'] = [row[0] for row in cursor.fetchall()]
               
               # Check if numpyarr column exists and its type
               if 'numpyarr' in info['table_columns']:
                   cursor.execute("SELECT typeof(numpyarr) FROM imag_with_seqs LIMIT 1")
                   result = cursor.fetchone()
                   info['numpyarr_type'] = result[0] if result else 'unknown'

               # Get available resolutions
               if 'resolution' in info['table_columns']:
                   cursor.execute("SELECT DISTINCT resolution FROM imag_with_seqs ORDER BY resolution")
                   info['available_resolutions'] = [row[0] for row in cursor.fetchall()]
               else:
                   info['available_resolutions'] = []
               
               # Get available datasets
               if 'dataset' in info['table_columns'] and 'hic_path' in info['table_columns']:
                   cursor.execute("SELECT DISTINCT dataset, hic_path FROM imag_with_seqs WHERE dataset IS NOT NULL AND hic_path IS NOT NULL ORDER BY dataset, hic_path")
                   dataset_results = cursor.fetchall()
                   
                   info['available_datasets'] = []
                   for dataset, hic_path in dataset_results:
                       # Extract identifier from hic_path
                       identifier = hic_path.split("/")[-1].rstrip(".hic") if hic_path else "unknown"
                       combined_id = f"{dataset}:{identifier}"
                       info['available_datasets'].append(combined_id)
               else:
                   info['available_datasets'] = []
               
               # Get detailed column information for debugging
               cursor.execute("PRAGMA table_info(imag_with_seqs)")
               column_details = cursor.fetchall()
               info['column_details'] = [
                   {
                       'cid': col[0],
                       'name': col[1], 
                       'type': col[2],
                       'notnull': col[3],
                       'default': col[4],
                       'pk': col[5]
                   } for col in column_details
               ]
           
           conn.close()
           return info
           
       except Exception as e:
           return {'error': str(e), 'file_path': db_path}
    
    @staticmethod
    def add_labels_column(db_path: str) -> bool:
        """Add labels column to imag_with_seqs table if it doesn't exist"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if column already exists
            cursor.execute("PRAGMA table_info(imag_with_seqs)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'labels' not in columns:
                # Add the labels column
                cursor.execute("ALTER TABLE imag_with_seqs ADD COLUMN labels TEXT DEFAULT ''")
                conn.commit()
                print(f"Added 'labels' column to imag_with_seqs table in {db_path}")
                return True
            else:
                print(f"'labels' column already exists in {db_path}")
                return True
                
        except Exception as e:
            print(f"Error adding labels column: {str(e)}")
            return False
        finally:
            conn.close()


class SampleSelectionDialog(QDialog):
    """Dialog for selecting samples from source database"""
    
    def __init__(self, db_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.db_info = db_info
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("Configure Sample Selection")
        self.setModal(True)
        self.resize(700, 600)
        
        layout = QVBoxLayout()
        
        # Database info
        info_label = QLabel(f"Database: {os.path.basename(self.db_info['file_path'])}")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Table info
        if self.db_info.get('has_target_table'):
            table_info = QLabel(f"Found 'imag_with_seqs' table with {self.db_info['table_count']} records")
            layout.addWidget(table_info)
            
            # Show column information
            columns_text = "Columns found: " + ", ".join(self.db_info['table_columns'])
            columns_label = QLabel(columns_text)
            columns_label.setWordWrap(True)
            columns_label.setStyleSheet("font-size: 10px; color: #888; margin: 5px 0;")
            layout.addWidget(columns_label)
            
            if self.db_info['key_id_range'][0] is not None:
                range_info = QLabel(f"Key ID range: {self.db_info['key_id_range'][0]} - {self.db_info['key_id_range'][1]}")
                layout.addWidget(range_info)
            
            # Labels column status
            if self.db_info['has_labels_column']:
                labels_status = QLabel("✓ Labels column exists")
                labels_status.setStyleSheet("color: #222222; font-weight: bold;")
            else:
                labels_status = QLabel("⚠ Labels column will be added")
                labels_status.setStyleSheet("color: #222222; font-weight: bold;")
            layout.addWidget(labels_status)
            
        else:
            error_label = QLabel("Error: No 'imag_with_seqs' table found in database!")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(error_label)
            
            # Show available tables
            if self.db_info.get('tables'):
                available_label = QLabel("Available tables: " + ", ".join(self.db_info['tables']))
                available_label.setWordWrap(True)
                available_label.setStyleSheet("font-size: 10px; margin: 5px 0;")
                layout.addWidget(available_label)
        
        # Sample count selection
        form_layout = QFormLayout()

        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setMinimum(1)
        self.sample_count_spin.setMaximum(self.db_info.get('table_count', 1))
        self.sample_count_spin.setValue(min(1000, self.db_info.get('table_count', 1)))
        form_layout.addRow("Number of samples:", self.sample_count_spin)
        
        # Random sampling option
        self.random_sample_check = QCheckBox("Random sampling")
        self.random_sample_check.setChecked(True)
        self.random_sample_check.setToolTip("If unchecked, will take first N samples ordered by key_id")
        form_layout.addRow("", self.random_sample_check)
        

        # Random seed input (only if random sampling)
        self.seed_spin = QSpinBox()
        self.seed_spin.setMinimum(0)
        self.seed_spin.setMaximum(999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setEnabled(self.random_sample_check.isChecked())
        form_layout.addRow("Random seed:", self.seed_spin)

        # Connect to enable/disable seed input
        self.random_sample_check.toggled.connect(self.seed_spin.setEnabled)


        # Add labels column option (if needed)
        if self.db_info.get('needs_labels_column'):
            self.add_labels_check = QCheckBox("Add labels column to source database")
            self.add_labels_check.setChecked(True)
            self.add_labels_check.setToolTip("This will modify the source database to add a labels column")
            form_layout.addRow("", self.add_labels_check)
        else:
            self.add_labels_check = None
        
        layout.addLayout(form_layout)

        # Filter selection mode
        filter_group = QGroupBox("Filter Selection Mode")
        filter_layout = QVBoxLayout()
        
        self.filter_mode_group = QButtonGroup()
        
        self.dumb_mode = QRadioButton("Dumb (ignore filters)")
        self.dumb_mode.setChecked(True)
        self.filter_mode_group.addButton(self.dumb_mode)
        filter_layout.addWidget(self.dumb_mode)
        
        self.smart_mode = QRadioButton("Smart (filter by resolution and dataset)")
        self.filter_mode_group.addButton(self.smart_mode)
        filter_layout.addWidget(self.smart_mode)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Resolution selection
        if self.db_info.get('available_resolutions'):
            resolution_group = QGroupBox("Resolution Selection")
            resolution_layout = QVBoxLayout()
            
            # Resolution checkboxes (initially disabled)
            self.resolution_checkboxes = {}
            checkbox_layout = QGridLayout()
            
            # Add "All" option
            self.all_resolutions_check = QCheckBox("All")
            self.all_resolutions_check.setEnabled(False)
            checkbox_layout.addWidget(self.all_resolutions_check, 0, 0)
            
            # Add individual resolution checkboxes
            resolutions = self.db_info['available_resolutions']
            for i, resolution in enumerate(resolutions):
                checkbox = QCheckBox(str(resolution))
                checkbox.setEnabled(False)
                self.resolution_checkboxes[resolution] = checkbox
                row = (i + 1) // 4
                col = (i + 1) % 4
                checkbox_layout.addWidget(checkbox, row, col)
            
            resolution_layout.addLayout(checkbox_layout)
            resolution_group.setLayout(resolution_layout)
            layout.addWidget(resolution_group)
        
        # Dataset selection
        if self.db_info.get('available_datasets'):
            dataset_group = QGroupBox("Dataset Selection")
            dataset_layout = QVBoxLayout()
            
            # Dataset checkboxes (initially disabled)
            self.dataset_checkboxes = {}
            dataset_checkbox_layout = QGridLayout()
            
            # Add "All" option
            self.all_datasets_check = QCheckBox("All")
            self.all_datasets_check.setEnabled(False)
            dataset_checkbox_layout.addWidget(self.all_datasets_check, 0, 0)
            
            # Add individual dataset checkboxes
            datasets = self.db_info['available_datasets']
            for i, dataset in enumerate(datasets):
                checkbox = QCheckBox(str(dataset))
                checkbox.setEnabled(False)
                self.dataset_checkboxes[dataset] = checkbox
                row = (i + 1) // 4
                col = (i + 1) % 4
                dataset_checkbox_layout.addWidget(checkbox, row, col)
            
            dataset_layout.addLayout(dataset_checkbox_layout)
            dataset_group.setLayout(dataset_layout)
            layout.addWidget(dataset_group)
        
        # Connect signals
        self.smart_mode.toggled.connect(self.on_filter_mode_changed)
        if hasattr(self, 'all_resolutions_check'):
            self.all_resolutions_check.toggled.connect(self.on_all_resolutions_changed)
        if hasattr(self, 'all_datasets_check'):
            self.all_datasets_check.toggled.connect(self.on_all_datasets_changed)

        # Warning about database modification
        if self.db_info.get('needs_labels_column'):
            warning_label = QLabel("⚠ Warning: This will modify your source database by adding a 'labels' column.")
            warning_label.setWordWrap(True)
            warning_label.setStyleSheet("color: #222222; font-weight: bold; padding: 10px; background-color: #3B4252; border-radius: 4px; margin: 10px 0;")
            layout.addWidget(warning_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(self.db_info.get('has_target_table', False))
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_selection_config(self) -> Dict[str, Any]:
        """Get the user's selection configuration"""
        config = {
            'sample_count': self.sample_count_spin.value(),
            'random_sampling': self.random_sample_check.isChecked(),
            'random_seed': self.seed_spin.value() if self.random_sample_check.isChecked() else 42,
            'add_labels_column': False,
            'filter_mode': 'smart' if hasattr(self, 'smart_mode') and self.smart_mode.isChecked() else 'dumb',
            'selected_resolutions': [],
            'selected_datasets': [],
            'filter_count': 0
        }
                
        # Handle filter selection if smart mode is enabled
        if hasattr(self, 'smart_mode') and self.smart_mode.isChecked():
            # Get selected resolutions
            if hasattr(self, 'all_resolutions_check'):
                if self.all_resolutions_check.isChecked():
                    config['selected_resolutions'] = self.db_info['available_resolutions']
                else:
                    config['selected_resolutions'] = [
                        res for res, checkbox in self.resolution_checkboxes.items() 
                        if checkbox.isChecked()
                    ]

            # Get selected datasets  
            if hasattr(self, 'all_datasets_check'):
                if self.all_datasets_check.isChecked():
                    config['selected_datasets'] = self.db_info['available_datasets']
                else:
                    config['selected_datasets'] = [
                        dataset for dataset, checkbox in self.dataset_checkboxes.items() 
                        if checkbox.isChecked()
                    ]
            
            # Calculate filter count (number of combinations)
            res_count = len(config['selected_resolutions']) if config['selected_resolutions'] else 1
            dataset_count = len(config['selected_datasets']) if config['selected_datasets'] else 1
            config['filter_count'] = res_count * dataset_count
        else:
            # In dumb mode, filter_count is effectively 1 (all data treated as one pool)
            config['filter_count'] = 1
        
        # Handle labels column addition
        if hasattr(self, 'add_labels_check') and self.add_labels_check:
            config['add_labels_column'] = self.add_labels_check.isChecked()
        
        return config
         
    def on_filter_mode_changed(self, checked):
        """Enable/disable filter selection based on mode"""
        if hasattr(self, 'all_resolutions_check'):
            self.all_resolutions_check.setEnabled(checked)
            for checkbox in self.resolution_checkboxes.values():
                checkbox.setEnabled(checked)
        
        if hasattr(self, 'all_datasets_check'):
            self.all_datasets_check.setEnabled(checked)
            for checkbox in self.dataset_checkboxes.values():
                checkbox.setEnabled(checked)

    def on_all_resolutions_changed(self, checked):
        """Handle 'All resolutions' checkbox toggle"""
        if checked:
            for checkbox in self.resolution_checkboxes.values():
                checkbox.setChecked(True)
    
    def on_all_datasets_changed(self, checked):
        """Handle 'All datasets' checkbox toggle"""
        if checked:
            for checkbox in self.dataset_checkboxes.values():
                checkbox.setChecked(True)


class SourceDatabaseProcessor(QThread):
    """Processes source database and creates sample indices"""
    
    progress_updated = pyqtSignal(int, str)
    processing_complete = pyqtSignal(int)  # source_db_id
    processing_error = pyqtSignal(str)
    
    def __init__(self, source_db_path: str, labeling_db_path: str, config: Dict[str, Any]):
        super().__init__()
        self.source_db_path = source_db_path
        self.labeling_db_path = labeling_db_path
        self.config = config
        self.source_db_id = None
    
    def run(self):
        """Process the source database"""
        try:
            self.process_source_database()
            self.processing_complete.emit(self.source_db_id)
        except Exception as e:
            self.processing_error.emit(str(e))
    
    def process_source_database(self):
        """Process the source database and create indices"""
        self.progress_updated.emit(0, "Connecting to databases...")
        
        # Add labels column to source database if needed
        if self.config.get('add_labels_column', False):
            self.progress_updated.emit(5, "Adding labels column to source database...")
            if not SourceDatabaseInspector.add_labels_column(self.source_db_path):
                raise Exception("Failed to add labels column to source database")
        
        # Connect to both databases
        labeling_conn = sqlite3.connect(self.labeling_db_path)
        source_conn = sqlite3.connect(self.source_db_path)
        
        try:
            labeling_cursor = labeling_conn.cursor()
            source_cursor = source_conn.cursor()
            
            # Create or update source database entry
            db_name = os.path.basename(self.source_db_path)
            labeling_cursor.execute("""
                INSERT OR REPLACE INTO source_databases 
                (name, file_path, description, last_accessed)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (db_name, self.source_db_path, f"Imported from {self.source_db_path}"))
            
            self.source_db_id = labeling_cursor.lastrowid
            
            # Get existing entry if it was a replacement
            if labeling_cursor.rowcount == 0:
                labeling_cursor.execute("SELECT id FROM source_databases WHERE file_path = ?", (self.source_db_path,))
                self.source_db_id = labeling_cursor.fetchone()[0]
            
            self.progress_updated.emit(10, "Reading source data...")
            
            # Get sample key_ids from source database
            key_ids = self.get_filtered_key_ids(source_cursor)
            total_samples = len(key_ids)
            
            self.progress_updated.emit(20, f"Processing {total_samples} samples...")
            
            # Create train/val/test split
            indices_dict = self.create_train_val_test_split(total_samples)
            
            # Clear existing sample indices for this source database
            labeling_cursor.execute("DELETE FROM sample_indices WHERE source_db_id = ?", (self.source_db_id,))
            
            # Insert sample indices
            for i, key_id in enumerate(key_ids):
                progress = 20 + int((i / total_samples) * 70)
                self.progress_updated.emit(progress, f"Processing sample {i+1}/{total_samples}")
                
                # Determine split type
                split_type = self.get_split_type(i, indices_dict)
                
                labeling_cursor.execute("""
                    INSERT INTO sample_indices 
                    (source_db_id, key_id, original_index, split_type, is_active)
                    VALUES (?, ?, ?, ?, 1)
                """, (self.source_db_id, key_id, i, split_type))
            
            # Update total samples count
            labeling_cursor.execute("""
                UPDATE source_databases 
                SET total_samples = ? 
                WHERE id = ?
            """, (total_samples, self.source_db_id))
            
            labeling_conn.commit()
            
            self.progress_updated.emit(100, "Processing complete!")
            
        finally:
            labeling_conn.close()
            source_conn.close()
    
    def get_filtered_key_ids(self, source_cursor):
        """Get key_ids based on the filter configuration - optimized single pass"""
        seed = self.config.get('random_seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        
        if self.config.get('filter_mode') == 'dumb':
            # Original logic unchanged
            if self.config['random_sampling']:
                source_cursor.execute("SELECT key_id FROM imag_with_seqs ORDER BY RANDOM() LIMIT ?", 
                                    (self.config['sample_count'],))
            else:
                source_cursor.execute("SELECT key_id FROM imag_with_seqs ORDER BY key_id LIMIT ?", 
                                    (self.config['sample_count'],))
            return [row[0] for row in source_cursor.fetchall()]
        
        else:
            # Smart filtering - single pass optimization
            selected_resolutions = self.config.get('selected_resolutions', [])
            selected_datasets = self.config.get('selected_datasets', [])
            
            if not selected_resolutions and not selected_datasets:
                source_cursor.execute("SELECT key_id FROM imag_with_seqs ORDER BY key_id")
                all_keys = [row[0] for row in source_cursor.fetchall()]
                if self.config['random_sampling']:
                    random.shuffle(all_keys)
                return all_keys[:self.config['sample_count']]
            
            # Build combination keys for quick lookup
            valid_combinations = set()
            for resolution in (selected_resolutions if selected_resolutions else [None]):
                for dataset_tuple in (selected_datasets if selected_datasets else [None]):
                    if dataset_tuple and resolution:
                        parts = dataset_tuple.split(":", 1)
                        if len(parts) == 2:
                            dataset, identifier = parts
                            valid_combinations.add((dataset, identifier, resolution))
            
            # Single query to get ALL relevant data at once
            self.progress_updated.emit(10, "Loading all data in single pass...")
            
            # Build WHERE clause for all combinations
            where_clauses = []
            params = []
            for dataset, identifier, resolution in valid_combinations:
                where_clauses.append("(dataset = ? AND hic_path LIKE ? AND resolution = ?)")
                params.extend([dataset, f"%/{identifier}.hic", resolution])
            
            if where_clauses:
                where_clause = " OR ".join(where_clauses)
                query = f"""
                    SELECT key_id, dataset, hic_path, resolution 
                    FROM imag_with_seqs 
                    WHERE {where_clause}
                    ORDER BY dataset, hic_path, resolution, key_id
                """
                source_cursor.execute(query, params)
                
                # Group results by combination
                combination_data = {}
                for key_id, dataset, hic_path, resolution in source_cursor.fetchall():
                    identifier = hic_path.split("/")[-1].rstrip(".hic")
                    combo_key = f"{dataset}:{identifier}@{resolution}"
                    
                    if combo_key not in combination_data:
                        combination_data[combo_key] = []
                    combination_data[combo_key].append(key_id)
                
                # Now shuffle and sample each combination
                self.combination_sampling_info = {}
                key_ids = []
                
                for combo_key, combo_key_ids in combination_data.items():
                    if self.config['random_sampling']:
                        combo_random = random.Random(seed + hash(combo_key))
                        combo_random.shuffle(combo_key_ids)
                    
                    self.combination_sampling_info[combo_key] = {
                        'all_key_ids_ordered': combo_key_ids,
                        'initial_count': min(self.config['sample_count'], len(combo_key_ids)),
                        'total_available': len(combo_key_ids),
                        'is_random': self.config['random_sampling'],
                        'seed': seed
                    }
                    
                    key_ids.extend(combo_key_ids[:self.config['sample_count']])
                
                # Save sampling info
                if hasattr(self, 'combination_sampling_info'):
                    import json
                    sampling_info_path = self.source_db_path + ".sampling_info.json"
                    with open(sampling_info_path, 'w') as f:
                        json.dump(self.combination_sampling_info, f)
                
                self.progress_updated.emit(100, f"Loaded {len(key_ids)} samples")
                return key_ids
            
            return []

    def create_train_val_test_split(self, total_size: int) -> Dict[str, List[int]]:
        """Create reproducible train/val/test split"""
        set_seed(42)
        
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Use all samples or subset of 50000
        subset_size = min(50000, total_size)
        subset_indices = indices[:subset_size]
        
        # Split ratios
        train_size = int(0.7 * subset_size)
        val_size = int(0.15 * subset_size)
        
        return {
            'train': subset_indices[:train_size],
            'val': subset_indices[train_size:train_size + val_size],
            'test': subset_indices[train_size + val_size:]
        }
    
    def get_split_type(self, index: int, indices_dict: Dict[str, List[int]]) -> str:
        """Get split type for given index"""
        if index in indices_dict['train']:
            return 'train'
        elif index in indices_dict['val']:
            return 'val'
        elif index in indices_dict['test']:
            return 'test'
        else:
            return 'train'  # Default


class ProcessingDialog(QDialog):
    """Dialog showing processing progress"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.worker = None
    
    def setup_ui(self):
        self.setWindowTitle("Processing Source Database")
        self.setModal(True)
        self.resize(400, 150)
        
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Preparing processing...")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_processing)
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
    
    def start_processing(self, source_db_path: str, labeling_db_path: str, config: Dict[str, Any]):
        """Start the processing"""
        self.worker = SourceDatabaseProcessor(source_db_path, labeling_db_path, config)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.processing_complete.connect(self.processing_finished)
        self.worker.processing_error.connect(self.processing_failed)
        self.worker.start()
    
    @pyqtSlot(int, str)
    def update_progress(self, percentage: int, status: str):
        """Update progress display"""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(status)
    
    @pyqtSlot(int)
    def processing_finished(self, source_db_id: int):
        """Handle successful processing"""
        self.source_db_id = source_db_id
        self.accept()
    
    @pyqtSlot(str)
    def processing_failed(self, error: str):
        """Handle processing error"""
        QMessageBox.critical(self, "Processing Error", f"Failed to process database: {error}")
        self.reject()
    
    def cancel_processing(self):
        """Cancel the processing"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        self.reject()


class SplashScreen(QMainWindow):
    """Splash screen with menu options"""
    
    dataset_selected = pyqtSignal(int)  # source_db_id
    
    def __init__(self):
        super().__init__()
        self.labeling_db_path = "hic_labeling.db"
        self.setup_labeling_database()
        self.setup_ui()
    
    def setup_labeling_database(self):
        """Initialize the labeling SQLite database"""
        conn = sqlite3.connect(self.labeling_db_path)
        cursor = conn.cursor()
        
        try:
            # Create tables
            for sql in LabelingDatabaseSchema.CREATE_TABLES:
                cursor.execute(sql)
            
            # Insert default label categories if they don't exist
            for name, desc, color, order in LabelingDatabaseSchema.DEFAULT_CATEGORIES:
                cursor.execute("""
                    INSERT OR IGNORE INTO label_categories (name, description, color, sort_order)
                    VALUES (?, ?, ?, ?)
                """, (name, desc, color, order))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def setup_ui(self):
        self.setWindowTitle("HiC Image Labeling Tool - SQLite Source Edition")
        self.setGeometry(300, 300, 700, 600)
        
        # Set styling
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2E3440, stop:1 #3B4252);
            }
            QPushButton {
                background-color: #5E81AC;
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
            QPushButton:pressed {
                background-color: #4C566A;
            }
            QLabel {
                color: #222222;
                font-size: 18px;
                font-weight: bold;
            }
            QTableWidget {
                background-color: #3B4252;
                color: #ECEFF4;
                gridline-color: #4C566A;
                border: 1px solid #4C566A;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #5E81AC;
            }
            QHeaderView::section {
                background-color: #4C566A;
                color: #ECEFF4;
                padding: 8px;
                border: 1px solid #2E3440;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        central_widget.setLayout(layout)
        
        # Title
        title = QLabel("HiC Image Labeling Tool")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("SQLite Source Database Edition")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #D8DEE9; margin-bottom: 20px;")
        layout.addWidget(subtitle)
        
        # Add new source database button
        self.add_source_btn = QPushButton("ADD SOURCE DATABASE")
        self.add_source_btn.setToolTip("Add a SQLite3 database containing HiC image data")
        self.add_source_btn.clicked.connect(self.add_source_database)
        layout.addWidget(self.add_source_btn)
        
        # Source databases table
        sources_label = QLabel("Source Databases:")
        sources_label.setStyleSheet("font-size: 16px; color: #ECEFF4; margin-top: 20px;")
        layout.addWidget(sources_label)
        
        self.sources_table = QTableWidget()
        self.sources_table.setColumnCount(6)
        self.sources_table.setHorizontalHeaderLabels([
            "Name", "File Path", "Total Samples", "Labeled", "Progress", "Last Accessed"
        ])
        
        # Set column widths
        header = self.sources_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        self.sources_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sources_table.doubleClicked.connect(self.open_source_database)
        
        layout.addWidget(self.sources_table)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("OPEN SELECTED")
        self.open_btn.clicked.connect(self.open_selected_database)
        self.open_btn.setEnabled(False)
        bottom_layout.addWidget(self.open_btn)
        
        self.refresh_btn = QPushButton("REFRESH")
        self.refresh_btn.clicked.connect(self.refresh_source_databases)
        bottom_layout.addWidget(self.refresh_btn)
        
        self.delete_btn = QPushButton("DELETE SELECTED")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #BF616A;
                color: white;
            }
            QPushButton:hover {
                background-color: #D08770;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_selected_database)
        self.delete_btn.setEnabled(False)
        bottom_layout.addWidget(self.delete_btn)
        
        bottom_layout.addStretch()
        
        self.quit_btn = QPushButton("QUIT")
        self.quit_btn.setStyleSheet("""
            QPushButton {
                background-color: #BF616A;
                color: white;
            }
            QPushButton:hover {
                background-color: #D08770;
            }
        """)
        self.quit_btn.clicked.connect(self.close)
        bottom_layout.addWidget(self.quit_btn)
        
        layout.addLayout(bottom_layout)
        
        # Connect table selection
        self.sources_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Load existing source databases
        self.refresh_source_databases()
    
    def add_source_database(self):
        """Add a new source database"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SQLite3 Source Database", "", 
            "SQLite files (*.db *.sqlite *.sqlite3);;All files (*)"
        )
        
        if not file_path:
            return
        
        # Inspect the database
        db_info = SourceDatabaseInspector.inspect_database(file_path)
        
        if 'error' in db_info:
            QMessageBox.critical(self, "Error", f"Failed to inspect database: {db_info['error']}")
            return
        
        if not db_info.get('has_target_table'):
            QMessageBox.critical(
                self, "Error", 
                f"The selected database does not contain an 'imag_with_seqs' table.\n"
                f"Available tables: {', '.join(db_info.get('tables', []))}\n"
                "Please select a database with HiC image data in the 'imag_with_seqs' table."
            )
            return
        
        # Show sample selection dialog
        selection_dialog = SampleSelectionDialog(db_info, self)
        if selection_dialog.exec_() != QDialog.Accepted:
            return
        
        config = selection_dialog.get_selection_config()
        
        # Process the database
        processing_dialog = ProcessingDialog(self)
        processing_dialog.start_processing(file_path, self.labeling_db_path, config)
        
        if processing_dialog.exec_() == QDialog.Accepted:
            self.refresh_source_databases()
            QMessageBox.information(
                self, "Success", 
                f"Successfully added source database with {config['sample_count']} samples per filter combination."
            )
    
    def refresh_source_databases(self):
        """Refresh the source databases table"""
        conn = sqlite3.connect(self.labeling_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT sd.id, sd.name, sd.file_path, sd.total_samples, 
                       COALESCE(COUNT(l.id), 0) as labeled_count, sd.last_accessed
                FROM source_databases sd
                LEFT JOIN sample_indices si ON sd.id = si.source_db_id AND si.is_active = 1
                LEFT JOIN labels l ON sd.id = l.source_db_id AND l.key_id = si.key_id
                GROUP BY sd.id, sd.name, sd.file_path, sd.total_samples, sd.last_accessed
                ORDER BY sd.last_accessed DESC
            """)
            
            rows = cursor.fetchall()
            
            self.sources_table.setRowCount(len(rows))
            
            for i, (db_id, name, file_path, total, labeled, last_accessed) in enumerate(rows):
                self.sources_table.setItem(i, 0, QTableWidgetItem(name))
                
                # Show truncated path for display
                display_path = file_path
                if len(display_path) > 50:
                    display_path = "..." + display_path[-47:]
                self.sources_table.setItem(i, 1, QTableWidgetItem(display_path))
                self.sources_table.item(i, 1).setToolTip(file_path)
                
                self.sources_table.setItem(i, 2, QTableWidgetItem(str(total)))
                self.sources_table.setItem(i, 3, QTableWidgetItem(str(labeled)))
                
                progress = f"{labeled/total*100:.1f}%" if total > 0 else "0%"
                self.sources_table.setItem(i, 4, QTableWidgetItem(progress))
                
                # Format date
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = last_accessed
                self.sources_table.setItem(i, 5, QTableWidgetItem(date_str))
                
                # Store db_id in the first item
                self.sources_table.item(i, 0).setData(Qt.UserRole, db_id)
                
                # Check if source file still exists
                if not os.path.exists(file_path):
                    for col in range(self.sources_table.columnCount()):
                        item = self.sources_table.item(i, col)
                        if item:
                            item.setBackground(QColor("#BF616A"))  # Red background for missing files
                            if col == 1:
                                item.setToolTip(f"File not found: {file_path}")
            
        finally:
            conn.close()
    
    def on_selection_changed(self):
        """Handle table selection changes"""
        has_selection = len(self.sources_table.selectedItems()) > 0
        self.open_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
    
    def open_source_database(self, index):
        """Open source database by double-click"""
        self.open_selected_database()
    
    def open_selected_database(self):
        """Open the selected source database for labeling"""
        current_row = self.sources_table.currentRow()
        if current_row >= 0:
            db_id = self.sources_table.item(current_row, 0).data(Qt.UserRole)
            if db_id:
                self.dataset_selected.emit(db_id)
                self.close()
    
    def delete_selected_database(self):
        """Delete the selected source database"""
        current_row = self.sources_table.currentRow()
        if current_row >= 0:
            db_id = self.sources_table.item(current_row, 0).data(Qt.UserRole)
            db_name = self.sources_table.item(current_row, 0).text()
            
            reply = QMessageBox.question(
                self, "Confirm Delete",
                f"Are you sure you want to delete '{db_name}' and all its labels?\n"
                "This action cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                conn = sqlite3.connect(self.labeling_db_path)
                cursor = conn.cursor()
                
                try:
                    # Delete labels first (foreign key constraint)
                    cursor.execute("DELETE FROM labels WHERE source_db_id = ?", (db_id,))
                    
                    # Delete sample indices
                    cursor.execute("DELETE FROM sample_indices WHERE source_db_id = ?", (db_id,))
                    
                    # Delete source database entry
                    cursor.execute("DELETE FROM source_databases WHERE id = ?", (db_id,))
                    
                    conn.commit()
                    self.refresh_source_databases()
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to delete database: {str(e)}")
                finally:
                    conn.close()


class ImageBin:
    """Represents a bin of images loaded in memory"""
    
    def __init__(self, bin_number: int, start_index: int, end_index: int):
        self.bin_number = bin_number
        self.start_index = start_index
        self.end_index = end_index
        self.images = {}  # key_id -> (pixmap, metadata)
        self.labels = {}  # key_id -> label (pending changes)
        self.is_loaded = False
    
    def contains_index(self, index: int) -> bool:
        """Check if this bin contains the given index"""
        return self.start_index <= index <= self.end_index
    
    def get_key_id_for_index(self, index: int, sample_indices: List[Tuple]) -> int:
        """Get key_id for the given index"""
        if not self.contains_index(index):
            raise ValueError(f"Index {index} not in bin {self.bin_number}")
        return sample_indices[index][0]
    
    def has_image(self, key_id: int) -> bool:
        """Check if image is loaded for given key_id"""
        return key_id in self.images
    
    def add_image(self, key_id: int, pixmap: QPixmap, metadata: dict):
        """Add image to bin"""
        self.images[key_id] = (pixmap, metadata)
    
    def get_image(self, key_id: int) -> Tuple[QPixmap, dict]:
        """Get image from bin"""
        return self.images[key_id]
    
    def set_label(self, key_id: int, label: str):
        """Set pending label for key_id"""
        self.labels[key_id] = label
    
    def get_label(self, key_id: int) -> Optional[str]:
        """Get pending label for key_id"""
        return self.labels.get(key_id)
    
    def get_pending_labels(self) -> Dict[int, str]:
        """Get all pending labels that need to be saved"""
        return self.labels.copy()
    
    def clear_pending_labels(self):
        """Clear pending labels after saving"""
        self.labels.clear()


class CombinationData:
    """Data for a specific dataset:identifier:resolution combination"""
    
    def __init__(self, combination_key: str):
        self.combination_key = combination_key
        self.sample_indices = []  # List of (index, key_id, split_type) tuples
        self.label_counts = {
            "Strong Positive": 0,
            "Weak Positive": 0, 
            "Weak Negative": 0,
            "Negative": 0,
            "Noise": 0
        }
        self.target_count = 200  # Target per label
        self.is_bypassed = False
        self.initial_sample_count = 0  # Track initial request
        self.expanded_count = 0  # Track how many times we expanded
        self.all_available_key_ids = []  # Pre-seeded complete sequence
        self.loaded_key_ids = set()  # Currently loaded key_ids
        self.is_random = False  # Whether random sampling was used
        self.total_available = 0  # Total samples available in database


class GridDataManager:
    """Grid-based data manager with 25 images per bin"""
    
    BIN_SIZE = 25  # 5x5 grid

    def __init__(self, source_db_id: int, labeling_db_path: str):
        self.source_db_id = source_db_id
        self.labeling_db_path = labeling_db_path
        self.source_db_path = None
        self.sample_indices = []
        self.combinations = {}  # combination_key -> CombinationData
        self.current_combination = None
        self.current_index = 0
        
        # Binning system
        self.bins = {}  # bin_number -> ImageBin
        self.current_bin = None
        self.loaded_bins = set()  # Track which bins are loaded
        self.pending_saves = {}  # key_id -> label (all pending changes)
        
        self.load_source_info()
        self.load_sample_indices()
        self.total_bins = (len(self.sample_indices) + self.BIN_SIZE - 1) // self.BIN_SIZE
        
        self.setup_progress_tracking()

    def setup_progress_tracking(self):
        """Initialize progress tracking - optimized to use saved info"""
        self.combinations = {}
        
        # Load sampling info with pre-seeded sequences
        sampling_info_path = self.source_db_path + ".sampling_info.json"
        sampling_info = {}
        if os.path.exists(sampling_info_path):
            try:
                with open(sampling_info_path, 'r') as f:
                    sampling_info = json.load(f)
            except:
                pass
        
        if not sampling_info:
            # Fallback to creating a default combination
            self.combinations["default"] = CombinationData("default")
            combo = self.combinations["default"]
            combo.sample_indices = [(i, kid, st) for i, (kid, st) in enumerate(self.sample_indices)]
            combo.loaded_key_ids = set([kid for kid, _ in self.sample_indices])
            return
        
        # Use the saved info directly - no database queries needed
        for combination_key, info in sampling_info.items():
            self.combinations[combination_key] = CombinationData(combination_key)
            combo = self.combinations[combination_key]
            
            combo.all_available_key_ids = info['all_key_ids_ordered']
            combo.initial_sample_count = info['initial_count']
            combo.is_random = info['is_random']
            combo.total_available = info['total_available']
            
            # The first N items are what's currently loaded
            loaded_count = min(combo.initial_sample_count, len(combo.all_available_key_ids))
            combo.loaded_key_ids = set(combo.all_available_key_ids[:loaded_count])
            
            # Map to sample_indices efficiently
            key_id_to_index = {kid: i for i, (kid, _) in enumerate(self.sample_indices)}
            
            for key_id in combo.all_available_key_ids[:loaded_count]:
                if key_id in key_id_to_index:
                    idx = key_id_to_index[key_id]
                    combo.sample_indices.append((idx, key_id, self.sample_indices[idx][1]))

    def load_source_info(self):
        """Load source database information"""
        conn = sqlite3.connect(self.labeling_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT file_path, name FROM source_databases WHERE id = ?
            """, (self.source_db_id,))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Source database with id {self.source_db_id} not found")
            
            self.source_db_path, self.source_db_name = result
            
            # Check if source file exists
            if not os.path.exists(self.source_db_path):
                raise FileNotFoundError(f"Source database file not found: {self.source_db_path}")
            
        finally:
            conn.close()
    
    def load_sample_indices(self):
        """Load sample indices for this source database"""
        conn = sqlite3.connect(self.labeling_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT key_id, split_type FROM sample_indices 
                WHERE source_db_id = ? AND is_active = 1
                ORDER BY original_index
            """, (self.source_db_id,))
            
            self.sample_indices = cursor.fetchall()
            
            if not self.sample_indices:
                raise ValueError("No active sample indices found for this source database")
            
            print(f"Loaded {len(self.sample_indices)} sample indices")
            
        finally:
            conn.close()

    def get_bin_number(self, index: int) -> int:
        """Get bin number for given index"""
        return index // self.BIN_SIZE
    
    def get_bin_range(self, bin_number: int) -> Tuple[int, int]:
        """Get start and end indices for a bin"""
        start_index = bin_number * self.BIN_SIZE
        end_index = min(start_index + self.BIN_SIZE - 1, len(self.sample_indices) - 1)
        return start_index, end_index
    
    def get_bins_to_load(self, current_bin: int) -> List[int]:
        """Get list of bins to keep in memory (current + 2 neighbors in each direction)"""
        bins_to_load = [current_bin]
        
        for offset in [1, 2]:
            if current_bin - offset >= 0:
                bins_to_load.append(current_bin - offset)
            if current_bin + offset < self.total_bins:
                bins_to_load.append(current_bin + offset)
        
        return sorted(bins_to_load)
    
    def ensure_bins_loaded(self, index: int):
        """Ensure necessary bins are loaded for the given index"""
        target_bin = self.get_bin_number(index)
        bins_needed = self.get_bins_to_load(target_bin)
        
        # Remove bins we no longer need
        bins_to_remove = set(self.loaded_bins) - set(bins_needed)
        for bin_num in bins_to_remove:
            if bin_num in self.bins:
                print(f"Unloading bin {bin_num}")
                del self.bins[bin_num]
                self.loaded_bins.discard(bin_num)
        
        # Load bins we need that aren't loaded
        for bin_num in bins_needed:
            if bin_num not in self.loaded_bins:
                self.load_bin(bin_num)
        
        self.current_bin = target_bin

    def load_bin(self, bin_number: int):
        """Load a specific bin into memory"""
        start_index, end_index = self.get_bin_range(bin_number)
        print(f"Loading bin {bin_number} (indices {start_index}-{end_index})")
        
        # Create bin object
        bin_obj = ImageBin(bin_number, start_index, end_index)
        
        # Get key_ids for this bin
        key_ids_in_bin = [self.sample_indices[i][0] for i in range(start_index, end_index + 1)]
        
        # Load images from database for this bin
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            # Get available columns
            source_cursor.execute("PRAGMA table_info(imag_with_seqs)")
            available_columns = [col[1] for col in source_cursor.fetchall()]
            
            # Build query with available columns
            base_columns = ['key_id']
            optional_columns = ['resolution', 'viewing_vmax', 'numpyarr', 'dimensions', 
                              'hic_path', 'PUB_ID', 'dataset', 'condition', 'labels']
            
            query_columns = base_columns + [col for col in optional_columns if col in available_columns]
            
            # Load all images in this bin with a single query
            placeholders = ','.join(['?' for _ in key_ids_in_bin])
            query = f"SELECT {', '.join(query_columns)} FROM imag_with_seqs WHERE key_id IN ({placeholders})"
            source_cursor.execute(query, key_ids_in_bin)
            
            rows = source_cursor.fetchall()
            
            for row in rows:
                # Create data dictionary
                data_dict = dict(zip(query_columns, row))
                
                key_id = data_dict.get('key_id')
                numpyarr_blob = data_dict.get('numpyarr')
                
                if numpyarr_blob is None:
                    continue
                
                # Get metadata
                resolution = data_dict.get('resolution', 2000)
                viewing_vmax = data_dict.get('viewing_vmax', 1.0)
                dimensions = data_dict.get('dimensions')
                
                # Find the index for this key_id
                sample_index = None
                split_type = None
                for i in range(start_index, end_index + 1):
                    if self.sample_indices[i][0] == key_id:
                        sample_index = i
                        split_type = self.sample_indices[i][1]
                        break
                
                if sample_index is None:
                    continue
                
                # Deserialize and convert to pixmap
                try:
                    numpyarr = self.deserialize_numpy_array(numpyarr_blob)
                    
                    if dimensions is None:
                        dimensions = numpyarr.shape[0] if len(numpyarr.shape) >= 2 else len(numpyarr)
                    
                    pixmap = self.array_to_pixmap(numpyarr, viewing_vmax)
                    
                    # Create metadata
                    metadata = {
                        'key_id': key_id,
                        'resolution': resolution,
                        'viewing_vmax': viewing_vmax,
                        'dimensions': dimensions,
                        'hic_path': data_dict.get('hic_path', ''),
                        'pub_id': data_dict.get('PUB_ID', ''),
                        'dataset': data_dict.get('dataset', ''),
                        'condition': data_dict.get('condition', ''),
                        'current_label': data_dict.get('labels', ''),
                        'split_type': split_type,
                        'index': sample_index,
                        'total_samples': len(self.sample_indices),
                        'available_columns': available_columns,
                        'bin_number': bin_number
                    }
                    
                    bin_obj.add_image(key_id, pixmap, metadata)
                    
                except Exception as e:
                    print(f"Error processing key_id {key_id}: {str(e)}")
                    continue
        
        finally:
            source_conn.close()
        
        # Store bin and mark as loaded
        self.bins[bin_number] = bin_obj
        self.loaded_bins.add(bin_number)
        bin_obj.is_loaded = True
        
        print(f"Loaded bin {bin_number} with {len(bin_obj.images)} images")
    
    def get_grid_images(self, start_index: int) -> List[Tuple[QPixmap, dict]]:
        """Get up to 25 images starting from start_index"""
        if not self.current_combination:
            return []
        
        images = []
        combo_samples = self.current_combination.sample_indices
        
        for i in range(start_index, min(start_index + 25, len(combo_samples))):
            actual_index, key_id, _ = combo_samples[i]
            
            # Ensure bin is loaded
            bin_number = self.get_bin_number(actual_index)
            if bin_number not in self.bins:
                self.load_bin(bin_number)
            
            bin_obj = self.bins[bin_number]
            if bin_obj.has_image(key_id):
                pixmap, metadata = bin_obj.get_image(key_id)
                metadata['grid_index'] = i - start_index
                metadata['combo_index'] = i
                
                # OVERRIDE with pending label if it exists
                if key_id in self.pending_saves:
                    metadata['current_label'] = self.pending_saves[key_id]
                    print(f"Found pending label for key_id {key_id}: {self.pending_saves[key_id]}")
                
                images.append((pixmap, metadata))
        
        return images
        
    def update_labels(self, key_ids: List[int], label: str):
        """Update labels for multiple key_ids"""
        for key_id in key_ids:
            self.pending_saves[key_id] = label
            
            # Update in bins if loaded - INCLUDING METADATA
            for bin_obj in self.bins.values():
                if bin_obj.has_image(key_id):
                    bin_obj.set_label(key_id, label)
                    # Also update the cached metadata
                    if key_id in bin_obj.images:
                        pixmap, metadata = bin_obj.images[key_id]
                        metadata['current_label'] = label
                        bin_obj.images[key_id] = (pixmap, metadata)
        
        print(f"Queued {len(key_ids)} labels for saving: {label}")
    
    def clear_labels(self, key_ids: List[int]):
        """Clear labels for multiple key_ids (set to empty string)"""
        self.update_labels(key_ids, '')
    
    def get_pending_save_count(self) -> int:
        """Get number of pending label saves"""
        return len(self.pending_saves)
    
    def save_pending_labels(self) -> int:
        """Save all pending labels to database"""
        if not self.pending_saves:
            return 0
        
        saved_count = 0
        
        # Update source database if labels column exists
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()

        try:
            # Check if labels column exists
            source_cursor.execute("PRAGMA table_info(imag_with_seqs)")
            columns = [col[1] for col in source_cursor.fetchall()]
            
            if 'labels' in columns:
                for key_id, label in self.pending_saves.items():
                    source_cursor.execute("""
                        UPDATE imag_with_seqs 
                        SET labels = ? 
                        WHERE key_id = ?
                    """, (label, key_id))
                    saved_count += 1
                
                source_conn.commit()
                print(f"Saved {saved_count} labels to source database")
        except Exception as e:
            print(f"Error updating source database: {str(e)}")
        finally:
            source_conn.close()
        
        # Clear pending saves
        self.pending_saves.clear()
        
        # Clear pending labels from all loaded bins
        for bin_obj in self.bins.values():
            bin_obj.clear_pending_labels()
        
        return saved_count
    
    def deserialize_numpy_array(self, blob_data) -> np.ndarray:
        """Deserialize numpy array from blob data"""
        if blob_data is None:
            raise ValueError("No numpy array data found")
        
        try:
            # Try pickle first (most common format)
            return pickle.loads(blob_data)
        except:
            try:
                # Try numpy's native format
                return np.frombuffer(blob_data, dtype=np.float32).reshape(-1)
            except:
                # Try base64 decoding first
                try:
                    decoded = base64.b64decode(blob_data)
                    return pickle.loads(decoded)
                except:
                    raise ValueError("Could not deserialize numpy array data")
    
    def array_to_pixmap(self, array_data: np.ndarray, viewing_vmax: float) -> QPixmap:
        """Convert numpy array to QPixmap (120x120 for grid display)"""
        # Ensure 2D array
        if len(array_data.shape) == 3:
            if array_data.shape[2] == 1:
                array_data = array_data[:, :, 0]
            else:
                # Take first channel or convert to grayscale
                array_data = np.mean(array_data, axis=2)
        elif len(array_data.shape) == 1:
            # Try to reshape to square
            size = int(np.sqrt(len(array_data)))
            if size * size == len(array_data):
                array_data = array_data.reshape(size, size)
            else:
                raise ValueError(f"Cannot reshape 1D array of length {len(array_data)} to 2D")
        
        # Apply colormap
        cmap = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])

        # Normalize
        data_min = float(array_data.min())
        data_max = float(viewing_vmax)
        data_range = data_max - data_min
        
        if data_range > 0:
            arr_norm = (array_data - data_min) / data_range
        else:
            arr_norm = np.zeros_like(array_data)
        
        # Clip to [0, 1] range
        arr_norm = np.clip(arr_norm, 0, 1)
        
        # Apply colormap
        rgba = cmap(arr_norm)
        rgba = (rgba * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgba, mode="RGBA")
        
        # Resize for grid display (120x120)
        pil_image = pil_image.resize((120, 120), Image.NEAREST)
        
        # Convert to QPixmap
        # Save to temporary file (simpler than converting through QImage)
        temp_path = "temp_display_image.png"
        pil_image.save(temp_path)
        pixmap = QPixmap(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return pixmap
    
    def get_combination_list(self) -> List[str]:
        """Get list of all combination keys"""
        return list(self.combinations.keys())

    def set_current_combination(self, combination_index: int):
        """Set the current combination by index"""
        combo_keys = self.get_combination_list()
        if 0 <= combination_index < len(combo_keys):
            self.current_combination = self.combinations[combo_keys[combination_index]]
            # Reset bins when switching combinations
            self.bins.clear()
            self.loaded_bins.clear()
    
    def expand_current_combination(self, additional_count: int = None):
        """Expand the current combination with additional samples"""
        if not self.current_combination:
            return False
        
        combo = self.current_combination
        
        if additional_count is None:
            additional_count = combo.initial_sample_count
        
        # Current position in the pre-seeded sequence
        current_loaded = len(combo.loaded_key_ids)
        
        # Check if we can expand
        if current_loaded >= len(combo.all_available_key_ids):
            print(f"No more samples available for {combo.combination_key}")
            return False
        
        # Get next batch from pre-seeded sequence
        end_index = min(current_loaded + additional_count, len(combo.all_available_key_ids))
        next_samples = combo.all_available_key_ids[current_loaded:end_index]
        
        if not next_samples:
            return False
        
        print(f"Expanding {combo.combination_key} with {len(next_samples)} additional samples")
        
        # Add to global sample_indices and combination indices
        for key_id in next_samples:
            self.sample_indices.append((key_id, 'train'))
            new_index = len(self.sample_indices) - 1
            combo.sample_indices.append((new_index, key_id, 'train'))
            combo.loaded_key_ids.add(key_id)
        
        combo.expanded_count += 1
        
        # Clear bins to force reload
        self.bins.clear()
        self.loaded_bins.clear()
        
        return True


    def get_combination_list(self) -> List[str]:
        """Get list of all combination keys"""
        return list(self.combinations.keys())

    def set_current_combination(self, combination_index: int):
        """Set the current combination by index"""
        combo_keys = self.get_combination_list()
        if 0 <= combination_index < len(combo_keys):
            self.current_combination = self.combinations[combo_keys[combination_index]]
            # Reset bins when switching combinations
            self.bins.clear()
            self.loaded_bins.clear()

    def get_current_combination_progress(self) -> str:
        """Get progress text for current combination"""
        if not self.current_combination:
            return "No combination selected"
        
        combo = self.current_combination
        progress_lines = []
        
        for i, (label, count) in enumerate(combo.label_counts.items()):
            status = "✓" if count >= combo.target_count else ""
            progress_lines.append(f"{i}: {count}/{combo.target_count} {label} {status}")
        
        return "\n".join(progress_lines)

    def load_source_info(self):
        """Load source database information"""
        conn = sqlite3.connect(self.labeling_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT file_path, name FROM source_databases WHERE id = ?
            """, (self.source_db_id,))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Source database with id {self.source_db_id} not found")
            
            self.source_db_path, self.source_db_name = result
            
            # Check if source file exists
            if not os.path.exists(self.source_db_path):
                raise FileNotFoundError(f"Source database file not found: {self.source_db_path}")
            
        finally:
            conn.close()
    
    def update_progress_counts(self):
        """Update progress counts for all combinations"""
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            # Get counts for each combination and label
            for combination_key in self.progress_tracker.combinations.keys():
                dataset, identifier, resolution = combination_key.split(':')
                hic_path_pattern = f"%{identifier}.hic"
                
                # Count labels for this combination
                label_counts = {}
                for label in self.progress_tracker.labels:
                    source_cursor.execute("""
                        SELECT COUNT(*) FROM imag_with_seqs 
                        WHERE dataset = ? AND hic_path LIKE ? AND resolution = ? 
                        AND labels = ?
                    """, (dataset, hic_path_pattern, resolution, label))
                    
                    count = source_cursor.fetchone()[0]
                    label_counts[label] = count
                
                self.progress_tracker.update_progress(combination_key, label_counts)
        finally:
            source_conn.close()


    def update_active_indices(self):
        """Update active sample indices based on bypass settings"""
        self.active_sample_indices = []
        
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            for i, (key_id, split_type) in enumerate(self.sample_indices):
                # Get combination info for this sample
                source_cursor.execute("""
                    SELECT dataset, hic_path, resolution 
                    FROM imag_with_seqs 
                    WHERE key_id = ?
                """, (key_id,))
                
                result = source_cursor.fetchone()
                if result:
                    dataset, hic_path, resolution = result
                    identifier = hic_path.split("/")[-1].rstrip(".hic") if hic_path else "unknown"
                    combination_key = f"{dataset}:{identifier}:{resolution}"
                    
                    # Only include if not bypassed
                    if not self.progress_tracker.is_bypassed(combination_key):
                        self.active_sample_indices.append((i, key_id, split_type, combination_key))
        finally:
            source_conn.close()
    
    def handle_bypass_change(self, combination_key: str, bypassed: bool):
        """Handle when user changes bypass status"""
        self.progress_tracker.set_bypassed(combination_key, bypassed)
        self.update_active_indices()
        
        # Invalidate current bins since indices changed
        self.bins.clear()
        self.loaded_bins.clear()
        
        # Check if current position is still valid
        if self.active_sample_indices:
            # Find the closest valid index
            current_key_id = self.sample_indices[self.current_index][0] if self.current_index < len(self.sample_indices) else None
            
            # Try to find this key_id in active indices
            new_index = 0
            for i, (orig_idx, key_id, _, _) in enumerate(self.active_sample_indices):
                if key_id == current_key_id:
                    new_index = i
                    break
            
            self.current_index = new_index
        else:
            self.current_index = 0


    def load_sample_indices(self):
        """Load sample indices for this source database"""
        conn = sqlite3.connect(self.labeling_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT key_id, split_type FROM sample_indices 
                WHERE source_db_id = ? AND is_active = 1
                ORDER BY original_index
            """, (self.source_db_id,))
            
            self.sample_indices = cursor.fetchall()
            
            if not self.sample_indices:
                raise ValueError("No active sample indices found for this source database")
            
            print(f"Loaded {len(self.sample_indices)} sample indices")
            
        finally:
            conn.close()
    
    def get_bin_number(self, index: int) -> int:
        """Get bin number for given index"""
        return index // self.BIN_SIZE
    
    def get_bin_range(self, bin_number: int) -> Tuple[int, int]:
        """Get start and end indices for a bin"""
        start_index = bin_number * self.BIN_SIZE
        end_index = min(start_index + self.BIN_SIZE - 1, len(self.sample_indices) - 1)
        return start_index, end_index
        
    def get_bins_to_load(self, current_bin: int) -> List[int]:
        """Get list of bins to keep in memory (current + neighbors)"""
        bins_to_load = [current_bin]
        
        # With smaller bins, maybe load 2 bins in each direction instead of 1
        for offset in [1, 2]:
            if current_bin - offset >= 0:
                bins_to_load.append(current_bin - offset)
            if current_bin + offset < self.total_bins:
                bins_to_load.append(current_bin + offset)
        
        return sorted(bins_to_load)
        
    def ensure_bins_loaded(self, index: int):
        """Ensure necessary bins are loaded for the given index"""
        target_bin = self.get_bin_number(index)
        bins_needed = self.get_bins_to_load(target_bin)
        
        # Remove bins we no longer need
        bins_to_remove = set(self.loaded_bins) - set(bins_needed)
        for bin_num in bins_to_remove:
            if bin_num in self.bins:
                print(f"Unloading bin {bin_num}")
                del self.bins[bin_num]
                self.loaded_bins.discard(bin_num)
        
        # Load bins we need that aren't loaded
        for bin_num in bins_needed:
            if bin_num not in self.loaded_bins:
                self.load_bin(bin_num)
        
        self.current_bin = target_bin
    
    def jump_to_index(self, index: int):
        """Jump to a specific index and update bins accordingly"""
        if index < 0 or index >= len(self.sample_indices):
            raise ValueError(f"Index {index} out of range [0, {len(self.sample_indices)-1}]")
        
        self.current_index = index
        self.ensure_bins_loaded(index)
    
    def load_bin(self, bin_number: int):
        """Load a specific bin into memory"""
        start_index, end_index = self.get_bin_range(bin_number)
        print(f"Loading bin {bin_number} (indices {start_index}-{end_index})")
        
        # Create bin object
        bin_obj = ImageBin(bin_number, start_index, end_index)
        
        # Get key_ids for this bin
        key_ids_in_bin = [self.sample_indices[i][0] for i in range(start_index, end_index + 1)]
        
        # Load images from database for this bin
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            # Get available columns
            source_cursor.execute("PRAGMA table_info(imag_with_seqs)")
            available_columns = [col[1] for col in source_cursor.fetchall()]
            
            # Build query with available columns
            base_columns = ['key_id']
            optional_columns = ['resolution', 'viewing_vmax', 'numpyarr', 'dimensions', 
                              'hic_path', 'PUB_ID', 'dataset', 'condition', 'labels']
            
            query_columns = base_columns + [col for col in optional_columns if col in available_columns]
            
            # Load all images in this bin with a single query
            placeholders = ','.join(['?' for _ in key_ids_in_bin])
            query = f"SELECT {', '.join(query_columns)} FROM imag_with_seqs WHERE key_id IN ({placeholders})"
            source_cursor.execute(query, key_ids_in_bin)
            
            rows = source_cursor.fetchall()
            
            for row in rows:
                # Create data dictionary
                data_dict = dict(zip(query_columns, row))
                
                key_id = data_dict.get('key_id')
                numpyarr_blob = data_dict.get('numpyarr')
                
                if numpyarr_blob is None:
                    continue
                
                # Get metadata
                resolution = data_dict.get('resolution', 2000)
                viewing_vmax = data_dict.get('viewing_vmax', 1.0)
                dimensions = data_dict.get('dimensions')
                
                # Find the index for this key_id
                sample_index = None
                split_type = None
                for i in range(start_index, end_index + 1):
                    if self.sample_indices[i][0] == key_id:
                        sample_index = i
                        split_type = self.sample_indices[i][1]
                        break
                
                if sample_index is None:
                    continue
                
                # Deserialize and convert to pixmap
                try:
                    numpyarr = self.deserialize_numpy_array(numpyarr_blob)
                    
                    if dimensions is None:
                        dimensions = numpyarr.shape[0] if len(numpyarr.shape) >= 2 else len(numpyarr)
                    
                    pixmap = self.array_to_pixmap(numpyarr, viewing_vmax)
                    
                    # Create metadata
                    metadata = {
                        'key_id': key_id,
                        'resolution': resolution,
                        'viewing_vmax': viewing_vmax,
                        'dimensions': dimensions,
                        'hic_path': data_dict.get('hic_path', ''),
                        'pub_id': data_dict.get('PUB_ID', ''),
                        'dataset': data_dict.get('dataset', ''),
                        'condition': data_dict.get('condition', ''),
                        'current_label': data_dict.get('labels', ''),
                        'split_type': split_type,
                        'index': sample_index,
                        'total_samples': len(self.sample_indices),
                        'available_columns': available_columns,
                        'bin_number': bin_number
                    }
                    
                    bin_obj.add_image(key_id, pixmap, metadata)
                    
                except Exception as e:
                    print(f"Error processing key_id {key_id}: {str(e)}")
                    continue
        
        finally:
            source_conn.close()
        
        # Store bin and mark as loaded
        self.bins[bin_number] = bin_obj
        self.loaded_bins.add(bin_number)
        bin_obj.is_loaded = True
        
        print(f"Loaded bin {bin_number} with {len(bin_obj.images)} images")
    
    def get_image_data(self, index: int) -> Tuple[QPixmap, dict]:
        """Get image and metadata for display with binning"""
        if not self.current_combination:
            raise ValueError("No combination selected")
        
        if index >= len(self.current_combination.sample_indices):
            raise IndexError(f"Index {index} out of range")
        
        # Get the actual sample index and key_id from current combination
        actual_index, key_id, split_type = self.current_combination.sample_indices[index]
        
        # Ensure necessary bins are loaded using actual_index
        self.ensure_bins_loaded(actual_index)
        
        # Get the bin and verify image exists
        bin_number = self.get_bin_number(actual_index)
        
        if bin_number not in self.bins:
            raise ValueError(f"Bin {bin_number} not loaded")
        
        bin_obj = self.bins[bin_number]
        
        if not bin_obj.has_image(key_id):
            raise ValueError(f"Image for key_id {key_id} not found in bin {bin_number}")
        
        pixmap, metadata = bin_obj.get_image(key_id)
        
        # Update metadata with both indices for clarity
        metadata['display_index'] = index  # Index within current combination (what user sees)
        metadata['global_index'] = actual_index  # Index in global sample_indices (for binning)
        metadata['combination_key'] = self.current_combination.combination_key
        
        # Check for pending label
        pending_label = self.pending_saves.get(key_id)
        if pending_label:
            metadata['pending_label'] = pending_label
        
        return pixmap, metadata

    def get_current_label(self, index: int) -> str:
        """Get current label for the sample at given index"""
        if not self.current_combination:
            return ""
            
        if index >= len(self.current_combination.sample_indices):
            return ""
        
        _, key_id, _ = self.current_combination.sample_indices[index]
        
        # Check pending changes first
        if key_id in self.pending_saves:
            return self.pending_saves[key_id]
        
        # Check if image is loaded in memory
        actual_index = self.current_combination.sample_indices[index][0]
        bin_number = self.get_bin_number(actual_index)
        if bin_number in self.bins:
            bin_obj = self.bins[bin_number]
            if bin_obj.has_image(key_id):
                _, metadata = bin_obj.get_image(key_id)
                return metadata.get('current_label', '')
        
        # Fallback to database query
        return self.get_label_from_database(key_id)


    def get_label_from_database(self, key_id: int) -> str:
        """Get label from database"""
        # Try source database first
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            source_cursor.execute("PRAGMA table_info(imag_with_seqs)")
            columns = [col[1] for col in source_cursor.fetchall()]
            
            if 'labels' in columns:
                source_cursor.execute("SELECT labels FROM imag_with_seqs WHERE key_id = ?", (key_id,))
                result = source_cursor.fetchone()
                if result and result[0]:
                    return result[0]
        except:
            pass
        finally:
            source_conn.close()
        
        # Fallback to labeling database
        conn = sqlite3.connect(self.labeling_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT sd.id, sd.name, sd.file_path, sd.total_samples, 
                       COALESCE(COUNT(l.id), 0) as labeled_count, sd.last_accessed
                FROM source_databases sd
                LEFT JOIN sample_indices si ON sd.id = si.source_db_id AND si.is_active = 1
                LEFT JOIN labels l ON sd.id = l.source_db_id AND l.key_id = si.key_id
                GROUP BY sd.id, sd.name, sd.file_path, sd.total_samples, sd.last_accessed
                ORDER BY sd.last_accessed DESC
            """)
            
            rows = cursor.fetchall()
            
            self.sources_table.setRowCount(len(rows))
            
            for i, (db_id, name, file_path, total, labeled, last_accessed) in enumerate(rows):
                self.sources_table.setItem(i, 0, QTableWidgetItem(name))
                
                # Show truncated path for display
                display_path = file_path
                if len(display_path) > 50:
                    display_path = "..." + display_path[-47:]
                self.sources_table.setItem(i, 1, QTableWidgetItem(display_path))
                self.sources_table.item(i, 1).setToolTip(file_path)
                
                self.sources_table.setItem(i, 2, QTableWidgetItem(str(total)))
                self.sources_table.setItem(i, 3, QTableWidgetItem(str(labeled)))
                
                progress = f"{labeled/total*100:.1f}%" if total > 0 else "0%"
                self.sources_table.setItem(i, 4, QTableWidgetItem(progress))
                
                # Format date
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = last_accessed
                self.sources_table.setItem(i, 5, QTableWidgetItem(date_str))
                
                # Store db_id in the first item
                self.sources_table.item(i, 0).setData(Qt.UserRole, db_id)
                
                # Check if source file still exists
                if not os.path.exists(file_path):
                    for col in range(self.sources_table.columnCount()):
                        item = self.sources_table.item(i, col)
                        if item:
                            item.setBackground(QColor("#BF616A"))  # Red background for missing files
                            if col == 1:
                                item.setToolTip(f"File not found: {file_path}")
            
        finally:
            conn.close()



    def update_label(self, index: int, label: str):
        """Update label (store in pending saves, don't write to DB immediately)"""
        if not self.current_combination:
            raise ValueError("No combination selected")
            
        if index >= len(self.current_combination.sample_indices):
            raise IndexError(f"Index {index} out of range")
        
        actual_index, key_id, _ = self.current_combination.sample_indices[index]
        self.pending_saves[key_id] = label
        
        # Also update in the current bin if loaded
        bin_number = self.get_bin_number(actual_index)
        if bin_number in self.bins:
            self.bins[bin_number].set_label(key_id, label)
        
        # Update combination label count
        self.current_combination.label_counts[label] = self.current_combination.label_counts.get(label, 0) + 1
        
        print(f"Label for key_id {key_id} queued for saving: {label}")

    
    def get_pending_save_count(self) -> int:
        """Get number of pending label saves"""
        return len(self.pending_saves)
    
    def save_pending_labels(self) -> int:
        """Save all pending labels to database"""
        if not self.pending_saves:
            return 0
        
        saved_count = 0
        
        # Update source database if labels column exists
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()

        try:
            # Check if labels column exists
            source_cursor.execute("PRAGMA table_info(imag_with_seqs)")
            columns = [col[1] for col in source_cursor.fetchall()]
            
            if 'labels' in columns:
                for key_id, label in self.pending_saves.items():
                    source_cursor.execute("""
                        UPDATE imag_with_seqs 
                        SET labels = ? 
                        WHERE key_id = ?
                    """, (label, key_id))
                    saved_count += 1
                
                source_conn.commit()
                print(f"Saved {saved_count} labels to source database")
        except Exception as e:
            print(f"Error updating source database: {str(e)}")
        finally:
            source_conn.close()
        
        # Update labeling database
        conn = sqlite3.connect(self.labeling_db_path)
        cursor = conn.cursor()
        
        try:
            for key_id, label in self.pending_saves.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO labels 
                    (source_db_id, key_id, label, labeled_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (self.source_db_id, key_id, label))
            
            conn.commit()
            print(f"Saved {len(self.pending_saves)} labels to labeling database")
            
            # Clear pending saves
            self.pending_saves.clear()
            
            # Clear pending labels from all loaded bins
            for bin_obj in self.bins.values():
                bin_obj.clear_pending_labels()
            
        except Exception as e:
            print(f"Error saving to labeling database: {str(e)}")
            raise
        finally:
            conn.close()
        
        return saved_count
    
    def deserialize_numpy_array(self, blob_data) -> np.ndarray:
        """Deserialize numpy array from blob data"""
        if blob_data is None:
            raise ValueError("No numpy array data found")
        
        try:
            # Try pickle first (most common format)
            return pickle.loads(blob_data)
        except:
            try:
                # Try numpy's native format
                return np.frombuffer(blob_data, dtype=np.float32).reshape(-1)
            except:
                # Try base64 decoding first
                try:
                    decoded = base64.b64decode(blob_data)
                    return pickle.loads(decoded)
                except:
                    raise ValueError("Could not deserialize numpy array data")
    
    def array_to_pixmap(self, array_data: np.ndarray, viewing_vmax: float) -> QPixmap:
        """Convert numpy array to QPixmap"""
        # Ensure 2D array
        if len(array_data.shape) == 3:
            if array_data.shape[2] == 1:
                array_data = array_data[:, :, 0]
            else:
                # Take first channel or convert to grayscale
                array_data = np.mean(array_data, axis=2)
        elif len(array_data.shape) == 1:
            # Try to reshape to square
            size = int(np.sqrt(len(array_data)))
            if size * size == len(array_data):
                array_data = array_data.reshape(size, size)
            else:
                raise ValueError(f"Cannot reshape 1D array of length {len(array_data)} to 2D")
        
        # Apply colormap
        cmap = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])

        # Normalize
        data_min = float(array_data.min())
        data_max = float(viewing_vmax)
        data_range = data_max - data_min
        
        if data_range > 0:
            arr_norm = (array_data - data_min) / data_range
        else:
            arr_norm = np.zeros_like(array_data)
        
        # Clip to [0, 1] range
        arr_norm = np.clip(arr_norm, 0, 1)
        
        # Apply colormap
        rgba = cmap(arr_norm)
        rgba = (rgba * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgba, mode="RGBA")
        
        # Resize for display
        pil_image = pil_image.resize((400, 400), Image.NEAREST)
        
        # Convert to QPixmap
        # Save to temporary file (simpler than converting through QImage)
        temp_path = "temp_display_image.png"
        pil_image.save(temp_path)
        pixmap = QPixmap(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return pixmap
    
    def get_labeling_stats(self) -> Dict[str, int]:
        """Get labeling statistics"""
        # Get total samples
        total_samples = len(self.sample_indices)
        
        # Count labeled samples by checking both source database and labeling database
        labeled_count = 0
        split_stats_dict = {}
        
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()
        
        # Check if labels column exists in source database
        source_cursor.execute("PRAGMA table_info(imag_with_seqs)")
        columns = [col[1] for col in source_cursor.fetchall()]
        has_labels_column = 'labels' in columns
        
        try:
            for key_id, split_type in self.sample_indices:
                # Initialize split stats
                if split_type not in split_stats_dict:
                    split_stats_dict[split_type] = {'total': 0, 'labeled': 0}
                split_stats_dict[split_type]['total'] += 1
                
                # Check if labeled (including pending saves)
                is_labeled = False
                
                # Check pending saves first
                if key_id in self.pending_saves:
                    is_labeled = True
                elif has_labels_column:
                    # Check source database
                    source_cursor.execute("SELECT labels FROM imag_with_seqs WHERE key_id = ? AND labels IS NOT NULL AND labels != ''", (key_id,))
                    if source_cursor.fetchone():
                        is_labeled = True
                
                if not is_labeled:
                    # Check labeling database as fallback
                    conn = sqlite3.connect(self.labeling_db_path)
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            SELECT label FROM labels 
                            WHERE source_db_id = ? AND key_id = ? AND label IS NOT NULL AND label != ''
                        """, (self.source_db_id, key_id))
                        if cursor.fetchone():
                            is_labeled = True
                    finally:
                        conn.close()
                
                if is_labeled:
                    labeled_count += 1
                    split_stats_dict[split_type]['labeled'] += 1
        
        finally:
            source_conn.close()
        
        # Convert to list format for compatibility
        split_stats = [(split_type, stats['total'], stats['labeled']) 
                      for split_type, stats in split_stats_dict.items()]
        
        return {
            'total_samples': total_samples,
            'labeled_count': labeled_count,
            'unlabeled_count': total_samples - labeled_count,
            'progress_percent': (labeled_count / total_samples * 100) if total_samples > 0 else 0,
            'split_stats': split_stats,
            'pending_saves': len(self.pending_saves)
        }


    def batch_update_label_counts(self):
        """Batch update label counts for all combinations"""
        if not self.combinations:
            return
        
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            # Build a single query to get all label counts at once
            combination_conditions = []
            params = []
            
            for combination_key in self.combinations.keys():
                parts = combination_key.split('@')
                if len(parts) == 2:
                    dataset_id_parts = parts[0].split(':', 1)
                    if len(dataset_id_parts) == 2:
                        dataset, identifier = dataset_id_parts
                        resolution = parts[1]
                        
                        combination_conditions.append(
                            "(dataset = ? AND hic_path LIKE ? AND resolution = ?)"
                        )
                        params.extend([dataset, f"%/{identifier}.hic", resolution])
            
            if combination_conditions:
                where_clause = " OR ".join(combination_conditions)
                
                # Get all label counts in one query
                query = f"""
                    SELECT dataset, hic_path, resolution, labels, COUNT(*) as count
                    FROM imag_with_seqs
                    WHERE ({where_clause}) AND labels IS NOT NULL AND labels != ''
                    GROUP BY dataset, hic_path, resolution, labels
                """
                
                source_cursor.execute(query, params)
                
                # Process results and update combinations
                for dataset, hic_path, resolution, label, count in source_cursor.fetchall():
                    identifier = hic_path.split("/")[-1].rstrip(".hic")
                    combination_key = f"{dataset}:{identifier}@{resolution}"
                    
                    if combination_key in self.combinations and label in self.combinations[combination_key].label_counts:
                        self.combinations[combination_key].label_counts[label] = count
        
        finally:
            source_conn.close()


class DatasetCombinationButton(QPushButton):
    """Button showing dataset combination with label counts"""
    
    def __init__(self, combination_key: str, parent=None):
        super().__init__(parent)
        self.combination_key = combination_key
        self.label_counts = {
            "Strong Positive": 0,
            "Weak Positive": 0,
            "Weak Negative": 0,
            "Negative": 0,
            "Noise": 0
        }
        self.update_display()
        self.setMinimumHeight(100)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4C566A;
                color: #ECEFF4;
                border: 2px solid #5E81AC;
                border-radius: 8px;
                padding: 8px;
                text-align: left;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
                border-color: #81A1C1;
            }
            QPushButton:checked {
                background-color: #A3BE8C;
                color: #2E3440;
                border: 3px solid #A3BE8C;
            }
        """)
        self.setCheckable(True)
    
    def update_counts(self, counts: dict):
        """Update label counts and refresh display"""
        self.label_counts = counts
        self.update_display()
    
    def update_display(self):
        """Update button text with combination and counts"""
        # Parse combination key
        parts = self.combination_key.split('@')
        if len(parts) == 2:
            dataset_id = parts[0]
            resolution = parts[1]
            text = f"@{resolution}\n{dataset_id}\n\n"
        else:
            text = f"{self.combination_key}\n\n"
        
        # Add label counts
        for i, (label, count) in enumerate(self.label_counts.items(), 1):
            text += f"{i}: {count}  "
            if i == 3:  # Line break after 3 items
                text += "\n"
        
        self.setText(text)


class ImageGrid(QWidget):
    """5x5 grid of images with multi-selection support"""
    
    selection_changed = pyqtSignal(set)  # Set of selected grid indices
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_labels = []
        self.key_id_labels = []
        self.selected_indices = set()
        self.image_data = {}  # grid_index -> (key_id, metadata)
        
        self.is_dragging = False
        self.drag_start = None
        self.drag_rect = QRect()
        
        self.setup_ui()
        
        # Set fixed size for the grid to prevent resizing
        # 5 images * 120px + 4 gaps * 10px + margins
        grid_size = 5 * 120 + 4 * 10 + 20
        self.setFixedSize(grid_size, grid_size + 100)  # Extra height for key_id labels
    
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(main_layout)
        
        # Grid layout for images
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        grid_widget.setLayout(grid_layout)
        
        # Create 5x5 grid
        for row in range(5):
            for col in range(5):
                grid_index = row * 5 + col
                
                # Container for image and key_id
                container = QWidget()
                container.setFixedSize(120, 140)  # Fixed size container
                container_layout = QVBoxLayout()
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(2)
                container.setLayout(container_layout)
                
                # Image label - already fixed at 120x120
                img_label = ClickableImageLabel(grid_index)
                img_label.clicked.connect(self.on_image_clicked)
                self.image_labels.append(img_label)
                container_layout.addWidget(img_label)
                
                # Key ID label
                key_label = QLabel("")
                key_label.setAlignment(Qt.AlignCenter)
                key_label.setStyleSheet("font-size: 10px; color: #D8DEE9;")
                key_label.setFixedHeight(15)
                self.key_id_labels.append(key_label)
                container_layout.addWidget(key_label)
                
                grid_layout.addWidget(container, row, col)
        
        grid_widget.setFixedSize(660, 760)  # Fixed size for the grid widget
        main_layout.addWidget(grid_widget)
        
        # Enable mouse tracking for drag selection
        self.setMouseTracking(True)
        grid_widget.setMouseTracking(True)

    def load_images(self, images: List[Tuple[QPixmap, dict]]):
        """Load images into the grid"""
        self.image_data.clear()
        
        # Clear all images first
        for i in range(25):
            self.image_labels[i].clear()
            self.image_labels[i].setPixmap(QPixmap())  # Empty pixmap
            self.key_id_labels[i].setText("")
            self.image_labels[i].set_label_number("")  # Clear label
        
        # Load new images
        for pixmap, metadata in images:
            grid_index = metadata.get('grid_index', 0)
            if 0 <= grid_index < 25:
                self.image_labels[grid_index].setPixmap(pixmap)
                self.key_id_labels[grid_index].setText(f"ID: {metadata['key_id']}")
                self.image_data[grid_index] = (metadata['key_id'], metadata)
                
                # Set label number if image has a label
                current_label = metadata.get('current_label', '')
                if current_label and current_label != '':
                    self.image_labels[grid_index].set_label_number(current_label)

    def mousePressEvent(self, event: QMouseEvent):
        """Start drag selection or clear on empty click"""
        if event.button() == Qt.LeftButton:
            # Check if click is on empty space (not on any image)
            click_pos = event.pos()
            clicked_on_image = False
            
            for i, img_label in enumerate(self.image_labels):
                if i in self.image_data:
                    img_rect = img_label.geometry()
                    if img_rect.contains(click_pos):
                        clicked_on_image = True
                        break
            
            if not clicked_on_image:
                # Clicked on empty space - clear selection
                self.clear_selection()
            else:
                # Start drag selection
                self.is_dragging = True
                self.drag_start = event.pos()
                self.drag_rect = QRect(self.drag_start, self.drag_start)
                
                # Don't clear selection when starting drag with shift
                if not (event.modifiers() & Qt.ShiftModifier):
                    # Don't clear here either - let individual click handle it
                    pass
    
    def on_image_clicked(self, grid_index: int):
        """Handle image click with shift support"""
        if grid_index not in self.image_data:
            return
        
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.ShiftModifier:
            # Shift-click: toggle selection
            if grid_index in self.selected_indices:
                self.selected_indices.remove(grid_index)
                self.image_labels[grid_index].set_selected(False)
            else:
                self.selected_indices.add(grid_index)
                self.image_labels[grid_index].set_selected(True)
        elif modifiers == Qt.ControlModifier:
            # Ctrl-click: add to selection
            self.selected_indices.add(grid_index)
            self.image_labels[grid_index].set_selected(True)
        else:
            # Regular click: select only this one
            self.clear_selection()
            self.selected_indices.add(grid_index)
            self.image_labels[grid_index].set_selected(True)
        
        self.selection_changed.emit(self.selected_indices.copy())
    
    def mousePressEvent(self, event: QMouseEvent):
        """Start drag selection"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_start = event.pos()
            self.drag_rect = QRect(self.drag_start, self.drag_start)
            
            # Clear selection if not holding shift
            if not (event.modifiers() & Qt.ShiftModifier):
                self.clear_selection()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Update drag selection"""
        if self.is_dragging and self.drag_start:
            self.drag_rect = QRect(self.drag_start, event.pos()).normalized()
            self.update_drag_selection()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """End drag selection"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.drag_start = None
            self.drag_rect = QRect()
            self.update()
    
    def update_drag_selection(self):
        """Update selection based on drag rectangle"""
        for i, img_label in enumerate(self.image_labels):
            if i not in self.image_data:
                continue
            
            # Get image label geometry in parent coordinates
            img_rect = img_label.geometry()
            img_rect.moveTopLeft(img_label.parent().mapToParent(img_rect.topLeft()))
            
            # Check if image intersects with drag rectangle
            if self.drag_rect.intersects(img_rect):
                if i not in self.selected_indices:
                    self.selected_indices.add(i)
                    img_label.set_selected(True)
            elif not (QApplication.keyboardModifiers() & Qt.ShiftModifier):
                # Only deselect if not holding shift
                if i in self.selected_indices:
                    self.selected_indices.remove(i)
                    img_label.set_selected(False)
        
        self.selection_changed.emit(self.selected_indices.copy())
    
    def paintEvent(self, event):
        """Draw drag selection rectangle"""
        super().paintEvent(event)
        
        if self.is_dragging and not self.drag_rect.isNull():
            painter = QPainter(self)
            painter.setPen(QPen(QColor(136, 192, 208), 2))
            painter.setBrush(QColor(136, 192, 208, 50))
            painter.drawRect(self.drag_rect)
    
    def update_label_numbers(self, key_id_to_label: dict):
        """Update label numbers for specific images without reloading"""
        for grid_index, (key_id, metadata) in self.image_data.items():
            if key_id in key_id_to_label:
                label = key_id_to_label[key_id]
                self.image_labels[grid_index].set_label_number(label)


    def clear_selection(self):
        """Clear all selections"""
        for i in self.selected_indices:
            if i < len(self.image_labels):
                self.image_labels[i].set_selected(False)
        self.selected_indices.clear()
        self.selection_changed.emit(set())
    
    def select_all(self):
        """Select all images that have data"""
        self.clear_selection()
        for i in self.image_data.keys():
            self.selected_indices.add(i)
            self.image_labels[i].set_selected(True)
        self.selection_changed.emit(self.selected_indices.copy())
    
    def get_selected_key_ids(self) -> List[int]:
        """Get key_ids of selected images"""
        key_ids = []
        for i in self.selected_indices:
            if i in self.image_data:
                key_ids.append(self.image_data[i][0])
        return key_ids

class ClickableImageLabel(QLabel):
    """Clickable image label for grid selection"""
    
    clicked = pyqtSignal(int)  # grid_index
    
    def __init__(self, grid_index: int, parent=None):
        super().__init__(parent)
        self.grid_index = grid_index
        self.is_selected = False
        self.current_label = ""
        self.label_number = -1
        self.pixmap_cache = None  # Store the original pixmap
        self.setCursor(Qt.PointingHandCursor)
        self.setScaledContents(True)
        self.setFixedSize(120, 120)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #4C566A;
                background-color: #3B4252;
            }
        """)
    
    def setPixmap(self, pixmap):
        """Override setPixmap to cache the pixmap"""
        self.pixmap_cache = pixmap if not pixmap.isNull() else None
        super().setPixmap(pixmap)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.grid_index)
    
    def set_selected(self, selected: bool):
        """Set selection state and update visual"""
        self.is_selected = selected
        if selected:
            self.setStyleSheet("""
                QLabel {
                    border: 3px solid #88C0D0;
                    background-color: #5E81AC;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #4C566A;
                    background-color: #3B4252;
                }
            """)
    
    def set_label_number(self, label_text: str):
        """Set the label for this image"""
        self.current_label = label_text
        label_map = {
            "Strong Positive": 1,
            "Weak Positive": 2,
            "Weak Negative": 3,
            "Negative": 4,
            "Noise": 5
        }
        self.label_number = label_map.get(label_text, -1)
        self.update()
    
    def paintEvent(self, event):
        """Paint the image and overlay label number if present"""
        # Let QLabel draw the pixmap first
        super().paintEvent(event)
        
        # Then draw our overlay on top
        if self.label_number > 0 and self.pixmap_cache is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw semi-transparent green background square in top-left
            painter.fillRect(3, 3, 24, 24, QColor(34, 139, 34, 230))
            
            # Draw white number
            painter.setPen(QPen(Qt.white, 2))
            painter.setFont(QFont("Arial", 14, QFont.Bold))
            painter.drawText(QRect(3, 3, 24, 24), Qt.AlignCenter, str(self.label_number))
            
            painter.end()

class JumpLineEdit(QLineEdit):
    def focusOutEvent(self, event):
        self.clear()  # Clear when focus is lost
        super().focusOutEvent(event)







class KeyIdSearchWidget(QWidget):
    """Widget for searching and displaying images by key_id with labeling support"""
    
    def __init__(self, data_manager: GridDataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.current_key_ids = []
        self.current_page = 0
        self.images_per_page = 25
        self.labels = ["Strong Positive", "Weak Positive", "Weak Negative", "Negative", "Noise"]
        screen = QApplication.desktop().screenGeometry()
        self.needs_scrolling = screen.height() < 900
        self.setup_ui()

        self.setFocusPolicy(Qt.StrongFocus)

    def setup_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # LEFT PANEL - Search and Grid
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Search controls
        search_group = QGroupBox("Search by Key IDs")
        search_layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Enter key IDs as comma-separated values or ranges.\n"
                            "Examples: 1,2,3,4 or 1-10,20,30-35")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #D8DEE9; padding: 10px;")
        search_layout.addWidget(instructions)
        
        # Input field
        self.key_id_input = QLineEdit()
        self.key_id_input.setPlaceholderText("Enter key IDs (e.g., 1,2,3,4 or 1-10)")
        self.key_id_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                font-size: 12px;
                background-color: #3B4252;
                border: 2px solid #4C566A;
                border-radius: 4px;
                color: #ECEFF4;
            }
            QLineEdit:focus {
                border-color: #5E81AC;
            }
        """)
        self.key_id_input.returnPressed.connect(self.search_key_ids)
        search_layout.addWidget(self.key_id_input)        

        # Submit button
        self.submit_button = QPushButton("Search")
        self.submit_button.clicked.connect(self.search_key_ids)
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #5E81AC;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
        """)
        search_layout.addWidget(self.submit_button)
        
        # Results info
        self.results_label = QLabel("No search performed")
        self.results_label.setStyleSheet("color: #A3BE8C; padding: 10px;")
        search_layout.addWidget(self.results_label)
        
        search_group.setLayout(search_layout)
        left_layout.addWidget(search_group)
        
        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(False)
        grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.image_grid = ImageGrid()
        self.image_grid.selection_changed.connect(self.on_selection_changed)
        grid_scroll.setWidget(self.image_grid)

        if self.needs_scrolling:
            grid_scroll.setMaximumHeight(500)
            grid_scroll.setMaximumWidth(700)

        left_layout.addWidget(grid_scroll)

        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self.previous_page)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)
        
        nav_layout.addStretch()
        
        self.page_label = QLabel("Page 0 / 0")
        self.page_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.page_label)
        
        nav_layout.addStretch()
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.next_page)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)
        
        left_layout.addLayout(nav_layout)
        
        # Selection info
        self.selection_label = QLabel("No images selected")
        self.selection_label.setAlignment(Qt.AlignCenter)
        self.selection_label.setStyleSheet("padding: 10px; background-color: #3B4252; border-radius: 4px;")
        left_layout.addWidget(self.selection_label)
        
        # Status
        self.status_label = QLabel("Ready to search")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #D8DEE9; padding: 10px;")
        left_layout.addWidget(self.status_label)
        
        # RIGHT PANEL - Label controls
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Selection controls
        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.image_grid.select_all)
        selection_layout.addWidget(select_all_btn)
        
        clear_selection_btn = QPushButton("Clear Selection")
        clear_selection_btn.clicked.connect(self.image_grid.clear_selection)
        selection_layout.addWidget(clear_selection_btn)
        
        selection_group.setLayout(selection_layout)
        right_layout.addWidget(selection_group)
        
        # Label selection
        label_group = QGroupBox("Apply Label to Selection")
        label_layout = QVBoxLayout()
        
        self.label_group = QButtonGroup()
        self.radio_buttons = {}
        
        for i, label in enumerate(self.labels):
            radio = QRadioButton(f"({i+1}) {label}")
            self.radio_buttons[label] = radio
            self.label_group.addButton(radio, i)
            label_layout.addWidget(radio)
        
        # Add clear label radio
        self.clear_radio = QRadioButton("(C) Clear Label")
        self.radio_buttons["Clear"] = self.clear_radio
        self.label_group.addButton(self.clear_radio, len(self.labels))
        label_layout.addWidget(self.clear_radio)
        
        label_layout.addSpacing(10)
        
        # Write button
        self.write_button = QPushButton("APPLY TO SELECTED")
        self.write_button.clicked.connect(self.write_labels)
        self.write_button.setEnabled(False)
        self.write_button.setStyleSheet("""
            QPushButton {
                background-color: #A3BE8C;
                color: #2E3440;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
            }
            QPushButton:hover:enabled {
                background-color: #B8D1A6;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #6C7B95;
            }
        """)
        label_layout.addWidget(self.write_button)
        
        label_group.setLayout(label_layout)
        right_layout.addWidget(label_group)
        
        # Save controls
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout()
        
        self.save_button = QPushButton("SAVE TO DATABASE")
        self.save_button.clicked.connect(self.save_labels)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #EBCB8B;
                color: #2E3440;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #F7D794;
            }
        """)
        save_layout.addWidget(self.save_button)
        
        self.pending_saves_label = QLabel("Pending saves: 0")
        self.pending_saves_label.setStyleSheet("color: #D08770; font-weight: bold; padding: 10px;")
        self.pending_saves_label.setAlignment(Qt.AlignCenter)
        save_layout.addWidget(self.pending_saves_label)
        
        save_group.setLayout(save_layout)
        right_layout.addWidget(save_group)
        
        right_layout.addStretch()
        
        # Add panels to main layout
        layout.addWidget(left_panel, 2)
        layout.addWidget(right_panel, 1)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for labeling"""
        # Don't process if user is typing in the search box
        if self.key_id_input.hasFocus():
            super().keyPressEvent(event)
            return
        
        # Handle number keys for labels
        if event.key() == Qt.Key_1:
            self.radio_buttons[self.labels[0]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_2:
            self.radio_buttons[self.labels[1]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_3:
            self.radio_buttons[self.labels[2]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_4:
            self.radio_buttons[self.labels[3]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_5:
            self.radio_buttons[self.labels[4]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_C or event.key() == Qt.Key_Delete:
            self.clear_radio.setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_Return and not self.key_id_input.hasFocus():
            # Apply label with Enter key (if not in search box)
            if len(self.image_grid.selected_indices) > 0 and self.label_group.checkedButton():
                self.write_labels()
        elif event.key() == Qt.Key_Escape:
            # Clear selection with Escape
            self.image_grid.clear_selection()
        elif event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_A:
                # Select all with Ctrl+A
                self.image_grid.select_all()
            elif event.key() == Qt.Key_S:
                # Save with Ctrl+S
                self.save_labels()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse clicks to clear focus from search input"""
        # Clear focus from search input if clicking elsewhere
        widget = self.childAt(event.pos())
        if widget != self.key_id_input and not self.key_id_input.geometry().contains(event.pos()):
            self.key_id_input.clearFocus()
            self.setFocus()  # Give focus to the widget itself for keyboard shortcuts
        super().mousePressEvent(event)

    def on_selection_changed(self, selected_indices: set):
        """Handle selection change"""
        count = len(selected_indices)
        if count == 0:
            self.selection_label.setText("No images selected")
            self.write_button.setEnabled(False)
        elif count == 1:
            self.selection_label.setText("1 image selected")
            self.write_button.setEnabled(True)
        else:
            self.selection_label.setText(f"{count} images selected")
            self.write_button.setEnabled(True)
    
    def write_labels(self):
        """Apply selected label to selected images"""
        selected_button = self.label_group.checkedButton()
        if not selected_button:
            QMessageBox.warning(self, "Warning", "Please select a label before applying.")
            return
        
        key_ids = self.image_grid.get_selected_key_ids()
        if not key_ids:
            QMessageBox.warning(self, "Warning", "Please select images to label.")
            return
        
        if selected_button == self.clear_radio:
            # Clear labels
            self.data_manager.clear_labels(key_ids)
            self.status_label.setText(f"Cleared labels for {len(key_ids)} images")
            # Update display to remove label numbers
            self.image_grid.update_label_numbers({kid: "" for kid in key_ids})
        else:
            # Apply label
            label = selected_button.text().split(") ", 1)[1] if ") " in selected_button.text() else selected_button.text()
            self.data_manager.update_labels(key_ids, label)
            self.status_label.setText(f"Applied '{label}' to {len(key_ids)} images")
            # Update display to show new label numbers immediately
            self.image_grid.update_label_numbers({kid: label for kid in key_ids})
        
        # Update pending saves display
        self.update_pending_saves_display()
    
    def save_labels(self):
        """Save all pending labels to database"""
        pending_count = self.data_manager.get_pending_save_count()
        
        if pending_count == 0:
            QMessageBox.information(self, "Info", "No pending labels to save.")
            return
        
        try:
            saved_count = self.data_manager.save_pending_labels()
            self.status_label.setText(f"Saved {saved_count} labels to database")
            self.update_pending_saves_display()
            QMessageBox.information(self, "Success", f"Successfully saved {saved_count} labels.")
        except Exception as e:
            self.status_label.setText(f"Error saving labels: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save labels: {str(e)}")
    
    def update_pending_saves_display(self):
        """Update the pending saves display"""
        pending_count = self.data_manager.get_pending_save_count()
        self.pending_saves_label.setText(f"Pending saves: {pending_count}")
        
        # Change color based on count
        if pending_count >= 100:
            self.pending_saves_label.setStyleSheet("color: #BF616A; font-weight: bold; padding: 10px;")
        elif pending_count >= 50:
            self.pending_saves_label.setStyleSheet("color: #EBCB8B; font-weight: bold; padding: 10px;")
        else:
            self.pending_saves_label.setStyleSheet("color: #D08770; font-weight: bold; padding: 10px;")
    
    def parse_key_ids(self, input_text: str) -> List[int]:
        """Parse input text to extract key IDs"""
        key_ids = []
        
        # Split by comma
        parts = input_text.replace(' ', '').split(',')
        
        for part in parts:
            if '-' in part:
                # Handle range
                try:
                    start, end = part.split('-')
                    start_id = int(start)
                    end_id = int(end)
                    key_ids.extend(range(start_id, end_id + 1))
                except:
                    continue
            else:
                # Handle single ID
                try:
                    key_ids.append(int(part))
                except:
                    continue
        
        return key_ids
    
    def search_key_ids(self):
        """Search for images by key IDs"""
        input_text = self.key_id_input.text().strip()
        self.key_id_input.clearFocus()
        self.setFocus()

        if not input_text:
            QMessageBox.warning(self, "Warning", "Please enter at least one key ID")
            return
        
        # Parse key IDs
        key_ids = self.parse_key_ids(input_text)
        
        if not key_ids:
            QMessageBox.warning(self, "Warning", "No valid key IDs found in input")
            return
        
        # Store key IDs and reset pagination
        self.current_key_ids = key_ids
        self.current_page = 0
        
        # Update results label
        self.results_label.setText(f"Found {len(key_ids)} key IDs to search")
        
        # Load first page
        self.load_current_page()
    
    def load_current_page(self):
        """Load current page of search results"""
        if not self.current_key_ids:
            return
        
        # Calculate page boundaries
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.current_key_ids))
        
        # Get key IDs for this page
        page_key_ids = self.current_key_ids[start_idx:end_idx]
        
        # Load images from database
        images = self.load_images_by_key_ids(page_key_ids)
        
        # Display in grid with labels
        self.image_grid.load_images(images)
        
        # Update navigation
        total_pages = (len(self.current_key_ids) + self.images_per_page - 1) // self.images_per_page
        self.page_label.setText(f"Page {self.current_page + 1} / {total_pages}")
        
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < total_pages - 1)
        
        # Update status
        self.status_label.setText(f"Showing {len(page_key_ids)} images (IDs {start_idx + 1}-{end_idx} of {len(self.current_key_ids)})")
        
        # Update pending saves display
        self.update_pending_saves_display()
    
    def load_images_by_key_ids(self, key_ids: List[int]) -> List[Tuple[QPixmap, dict]]:
        """Load images for specific key IDs from database"""
        images = []
        
        source_conn = sqlite3.connect(self.data_manager.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            # Get available columns
            source_cursor.execute("PRAGMA table_info(imag_with_seqs)")
            available_columns = [col[1] for col in source_cursor.fetchall()]
            
            # Build query
            base_columns = ['key_id']
            optional_columns = ['resolution', 'viewing_vmax', 'numpyarr', 'dimensions', 
                              'hic_path', 'PUB_ID', 'dataset', 'condition', 'labels']
            
            query_columns = base_columns + [col for col in optional_columns if col in available_columns]
            
            # Query for each key_id
            for grid_index, key_id in enumerate(key_ids):
                if grid_index >= 25:  # Grid limit
                    break
                
                query = f"SELECT {', '.join(query_columns)} FROM imag_with_seqs WHERE key_id = ?"
                source_cursor.execute(query, (key_id,))
                row = source_cursor.fetchone()
                
                if row:
                    data_dict = dict(zip(query_columns, row))
                    numpyarr_blob = data_dict.get('numpyarr')
                    
                    if numpyarr_blob:
                        try:
                            # Deserialize numpy array
                            numpyarr = self.data_manager.deserialize_numpy_array(numpyarr_blob)
                            
                            # Convert to pixmap
                            viewing_vmax = data_dict.get('viewing_vmax', 1.0)
                            pixmap = self.array_to_pixmap_120(numpyarr, viewing_vmax)
                            
                            # Check for pending label first
                            current_label = self.data_manager.pending_saves.get(key_id, data_dict.get('labels', ''))
                            
                            # Create metadata
                            metadata = {
                                'key_id': key_id,
                                'resolution': data_dict.get('resolution', 'N/A'),
                                'viewing_vmax': viewing_vmax,
                                'dimensions': data_dict.get('dimensions', 'N/A'),
                                'hic_path': data_dict.get('hic_path', ''),
                                'pub_id': data_dict.get('PUB_ID', ''),
                                'dataset': data_dict.get('dataset', ''),
                                'condition': data_dict.get('condition', ''),
                                'current_label': current_label,  # Will show pending or saved label
                                'grid_index': grid_index
                            }
                            
                            images.append((pixmap, metadata))
                        except Exception as e:
                            print(f"Error processing key_id {key_id}: {str(e)}")
        finally:
            source_conn.close()
        
        return images
    
    def array_to_pixmap_120(self, array_data: np.ndarray, viewing_vmax: float) -> QPixmap:
        """Convert numpy array to 120x120 QPixmap for grid display"""
        # Ensure 2D array
        if len(array_data.shape) == 3:
            if array_data.shape[2] == 1:
                array_data = array_data[:, :, 0]
            else:
                array_data = np.mean(array_data, axis=2)
        elif len(array_data.shape) == 1:
            size = int(np.sqrt(len(array_data)))
            if size * size == len(array_data):
                array_data = array_data.reshape(size, size)
            else:
                raise ValueError(f"Cannot reshape 1D array of length {len(array_data)} to 2D")
        
        # Apply colormap
        cmap = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])
        
        # Normalize
        data_min = float(array_data.min())
        data_max = float(viewing_vmax)
        data_range = data_max - data_min
        
        if data_range > 0:
            arr_norm = (array_data - data_min) / data_range
        else:
            arr_norm = np.zeros_like(array_data)
        
        arr_norm = np.clip(arr_norm, 0, 1)
        
        # Apply colormap
        rgba = cmap(arr_norm)
        rgba = (rgba * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgba, mode="RGBA")
        
        # Resize for grid display
        pil_image = pil_image.resize((120, 120), Image.NEAREST)
        
        # Convert to QPixmap
        temp_path = "temp_search_image.png"
        pil_image.save(temp_path)
        pixmap = QPixmap(temp_path)
        
        try:
            os.remove(temp_path)
        except:
            pass
        
        return pixmap
    
    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.load_current_page()
    
    def next_page(self):
        """Go to next page"""
        total_pages = (len(self.current_key_ids) + self.images_per_page - 1) // self.images_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.load_current_page()




class ImageLabelingWindow(QMainWindow):
    """Main window for grid-based image labeling"""
    
    def __init__(self, data_manager: GridDataManager, splash_screen=None, pending_count=125):
        super().__init__()
        self.data_manager = data_manager
        self.splash_screen = splash_screen  # Store reference to splash screen
        self.current_page = 0
        self.labels = ["Strong Positive", "Weak Positive", "Weak Negative", "Negative", "Noise"]
        self.PENDING_COUNT = pending_count
        
        self.setWindowTitle(f"HiC Grid Labeling - {self.data_manager.source_db_name}")
        
        # Get screen dimensions and set window size accordingly
        screen = QApplication.desktop().screenGeometry()
        screen_height = screen.height()
        screen_width = screen.width()
        
        print(f"DEBUG: Screen dimensions: {screen_width}x{screen_height}")
        
        default_height = 900
        default_width = 1200
        
        if screen_height < default_height:
            window_height = int(screen_height * 7 / 8)
            self.needs_scrolling = True
            print(f"DEBUG: Small screen detected, setting height to 7/8: {window_height}")
        else:
            window_height = default_height
            self.needs_scrolling = False
            print(f"DEBUG: Using default height: {window_height}")
        
        window_width = min(default_width, int(screen_width * 7 / 8))
        
        # Center the window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        print(f"DEBUG: Initial window size: {window_width}x{window_height} at position ({x}, {y})")
        self.setGeometry(x, y, window_width, window_height)
        
        # Remove size constraints to allow resizing
        self.setMinimumSize(800, 600)  # Set reasonable minimum
        # Don't set maximum size to allow full resizing
        
        self.setup_ui_with_tabs()
        
        # Set initial combination and load first grid
        if self.data_manager.combinations:
            self.data_manager.set_current_combination(0)
            self.load_current_grid()
            self.update_all_combination_counts()


    def setup_ui_with_tabs(self):
        """Setup the user interface with tabs"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Set stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
            }
            QWidget {
                background-color: #2E3440;
                color: #ECEFF4;
            }
            QPushButton {
                background-color: #5E81AC;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #6C7B95;
            }
            QTabWidget::pane {
                border: 1px solid #4C566A;
                background-color: #2E3440;
            }
            QTabBar::tab {
                background-color: #3B4252;
                color: #D8DEE9;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #5E81AC;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #4C566A;
            }
            QRadioButton {
                color: #ECEFF4;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4C566A;
                border-radius: 8px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #ECEFF4;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top bar with back button
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar.setLayout(top_bar_layout)
        
        # Back button - discrete styling
        self.back_button = QPushButton("← Back to Menu")
        self.back_button.clicked.connect(self.return_to_menu)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #88C0D0;
                border: 1px solid #4C566A;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                text-align: left;
                max-width: 120px;
            }
            QPushButton:hover {
                background-color: #3B4252;
                border-color: #5E81AC;
            }
        """)
        top_bar_layout.addWidget(self.back_button)
        top_bar_layout.addStretch()
        
        main_layout.addWidget(top_bar)


        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab 1: Annotation (existing functionality)
        self.annotation_tab = QWidget()
        self.setup_annotation_tab()
        self.tab_widget.addTab(self.annotation_tab, "Annotation")
        
        # Tab 2: Key ID Search
        self.search_tab = KeyIdSearchWidget(self.data_manager)
        self.tab_widget.addTab(self.search_tab, "Key ID Search")
        
    def setup_annotation_tab(self):
        """Setup the annotation tab with existing functionality"""
        main_layout = QHBoxLayout()
        self.annotation_tab.setLayout(main_layout)
        
        # LEFT PANEL - Grid and navigation
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Dataset info
        self.info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout()
        self.dataset_info_label = QLabel()
        self.dataset_info_label.setWordWrap(True)
        info_layout.addWidget(self.dataset_info_label)
        self.info_group.setLayout(info_layout)
        left_layout.addWidget(self.info_group)
        
        # Image grid - make scrollable if screen is small
        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(False)
        grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.image_grid = ImageGrid()
        self.image_grid.selection_changed.connect(self.on_selection_changed)
        grid_scroll.setWidget(self.image_grid)
        
        if self.needs_scrolling:
            grid_scroll.setMaximumHeight(500)
            grid_scroll.setMaximumWidth(700)
        
        left_layout.addWidget(grid_scroll)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("◀ Previous Page")
        self.prev_button.clicked.connect(self.previous_page)
        nav_layout.addWidget(self.prev_button)
        
        nav_layout.addStretch()
        
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setFont(QFont("Arial", 12, QFont.Bold))
        nav_layout.addWidget(self.page_label)
        
        nav_layout.addStretch()
        
        self.next_button = QPushButton("Next Page ▶")
        self.next_button.clicked.connect(self.next_page)
        nav_layout.addWidget(self.next_button)
        
        left_layout.addLayout(nav_layout)
        
        # Selection info
        self.selection_label = QLabel("No images selected")
        self.selection_label.setAlignment(Qt.AlignCenter)
        self.selection_label.setStyleSheet("padding: 10px; background-color: #3B4252; border-radius: 4px;")
        left_layout.addWidget(self.selection_label)
        
        # RIGHT PANEL - Labels and controls
        right_panel = QWidget()
        right_panel.setFixedWidth(350)
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Selection controls
        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout()
        
        select_all_btn = QPushButton("Select All (Ctrl+A)")
        select_all_btn.clicked.connect(self.image_grid.select_all)
        select_all_btn.setShortcut("Ctrl+A")
        selection_layout.addWidget(select_all_btn)
        
        clear_selection_btn = QPushButton("Clear Selection (Escape)")
        clear_selection_btn.clicked.connect(self.image_grid.clear_selection)
        clear_selection_btn.setShortcut("Escape")
        selection_layout.addWidget(clear_selection_btn)
        
        selection_group.setLayout(selection_layout)
        right_layout.addWidget(selection_group)
        
        # Label selection
        label_group = QGroupBox("Apply Label to Selection")
        label_layout = QVBoxLayout()
        
        self.label_group = QButtonGroup()
        self.radio_buttons = {}
        
        for i, label in enumerate(self.labels):
            radio = QRadioButton(f"({i+1}) {label}")
            self.radio_buttons[label] = radio
            self.label_group.addButton(radio, i)
            label_layout.addWidget(radio)
        
        # Add clear label radio
        self.clear_radio = QRadioButton("(C/Delete) Clear Label")
        self.radio_buttons["Clear"] = self.clear_radio
        self.label_group.addButton(self.clear_radio, len(self.labels))
        label_layout.addWidget(self.clear_radio)
        
        label_layout.addSpacing(10)
        
        # Write button
        self.write_button = QPushButton("APPLY TO SELECTED")
        self.write_button.clicked.connect(self.write_labels)
        self.write_button.setEnabled(False)
        self.write_button.setStyleSheet("""
            QPushButton {
                background-color: #A3BE8C;
                color: #2E3440;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
            }
            QPushButton:hover:enabled {
                background-color: #B8D1A6;
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #6C7B95;
            }
        """)
        self.write_button.setShortcut("Return")
        label_layout.addWidget(self.write_button)
        
        label_group.setLayout(label_layout)
        right_layout.addWidget(label_group)
        
        # Save controls
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout()
        
        self.save_button = QPushButton("SAVE TO DATABASE")
        self.save_button.clicked.connect(self.save_labels)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #EBCB8B;
                color: #2E3440;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #F7D794;
            }
        """)
        self.save_button.setShortcut("Ctrl+S")
        save_layout.addWidget(self.save_button)
        
        self.pending_saves_label = QLabel("Pending saves: 0")
        self.pending_saves_label.setStyleSheet("color: #D08770; font-weight: bold; padding: 10px;")
        self.pending_saves_label.setAlignment(Qt.AlignCenter)
        save_layout.addWidget(self.pending_saves_label)
        
        save_group.setLayout(save_layout)
        right_layout.addWidget(save_group)
        
        # Combination buttons
        combo_group = QGroupBox("Dataset Combinations")
        combo_layout = QVBoxLayout()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        if self.needs_scrolling:
            scroll_area.setMaximumHeight(250)
        else:
            scroll_area.setMaximumHeight(400)
        
        scroll_widget = QWidget()
        self.combo_buttons_layout = QVBoxLayout()
        scroll_widget.setLayout(self.combo_buttons_layout)
        
        self.combo_button_group = QButtonGroup()
        self.combo_buttons = []
        
        for i, combo_key in enumerate(self.data_manager.get_combination_list()):
            button = DatasetCombinationButton(combo_key)
            button.clicked.connect(lambda checked, idx=i: self.on_combination_button_clicked(idx))
            self.combo_button_group.addButton(button, i)
            self.combo_buttons.append(button)
            self.combo_buttons_layout.addWidget(button)
        
        if self.combo_buttons:
            self.combo_buttons[0].setChecked(True)
        
        scroll_area.setWidget(scroll_widget)
        combo_layout.addWidget(scroll_area)
        
        self.combo_selector = QComboBox()
        self.combo_selector.currentIndexChanged.connect(self.on_combination_changed)
        for combo_key in self.data_manager.get_combination_list():
            self.combo_selector.addItem(combo_key)
        combo_layout.addWidget(QLabel("Or select from dropdown:"))
        combo_layout.addWidget(self.combo_selector)
        
        combo_group.setLayout(combo_layout)
        right_layout.addWidget(combo_group)
        
        right_layout.addStretch()
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #A3BE8C; font-weight: bold; padding: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)


    def return_to_menu(self):
        """Return to the database selection menu"""
        pending_count = self.data_manager.get_pending_save_count()
        
        if pending_count > 0:
            # Has pending saves - ask if they want to save first
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                f"You have {pending_count} unsaved labels.\n"
                "Do you want to save them before returning to the menu?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                try:
                    self.save_labels()
                    self.open_splash_screen()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save labels: {str(e)}")
            elif reply == QMessageBox.Discard:
                self.open_splash_screen()
            # else: Cancel - do nothing
        else:
            # No pending saves - just confirm
            reply = QMessageBox.question(
                self, "Return to Menu",
                "Are you sure you want to return to the database selection menu?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.open_splash_screen()

    def open_splash_screen(self):
        """Close current window and open splash screen"""
        if self.splash_screen:
            self.splash_screen.show()
            self.splash_screen.refresh_source_databases()
        else:
            # Create new splash screen if reference was lost
            from __main__ import SplashScreen
            splash = SplashScreen()
            splash.show()
        
        self.close()


    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_1:
            self.radio_buttons[self.labels[0]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_2:
            self.radio_buttons[self.labels[1]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_3:
            self.radio_buttons[self.labels[2]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_4:
            self.radio_buttons[self.labels[3]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_5:
            self.radio_buttons[self.labels[4]].setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        elif event.key() == Qt.Key_C or event.key() == Qt.Key_Delete:
            self.clear_radio.setChecked(True)
            if len(self.image_grid.selected_indices) > 0:
                self.write_labels()
        else:
            super().keyPressEvent(event)
    
    def on_combination_changed(self, index):
        """Handle dataset combination change"""
        self.data_manager.set_current_combination(index)
        self.current_page = 0
        self.load_current_grid()
    
    
    def on_selection_changed(self, selected_indices: set):
        """Handle selection change"""
        count = len(selected_indices)
        if count == 0:
            self.selection_label.setText("No images selected")
            self.write_button.setEnabled(False)
        elif count == 1:
            self.selection_label.setText("1 image selected")
            self.write_button.setEnabled(True)
        else:
            self.selection_label.setText(f"{count} images selected")
            self.write_button.setEnabled(True)
    
    def on_combination_button_clicked(self, index):
        """Handle combination button click"""
        self.data_manager.set_current_combination(index)
        self.current_page = 0
        self.combo_selector.setCurrentIndex(index)  # Keep dropdown in sync
        self.load_current_grid()
        self.update_all_combination_counts()

    def on_combination_changed(self, index):
        """Handle dataset combination change from dropdown"""
        self.data_manager.set_current_combination(index)
        self.current_page = 0
        # Update button selection
        if index < len(self.combo_buttons):
            self.combo_buttons[index].setChecked(True)
        self.load_current_grid()
        self.update_all_combination_counts()

    def update_all_combination_counts(self):
        """Update label counts for all combination buttons"""
        source_conn = sqlite3.connect(self.data_manager.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            for button in self.combo_buttons:
                combo_key = button.combination_key
                
                # Parse combination key
                parts = combo_key.split('@')
                if len(parts) == 2:
                    dataset_id_parts = parts[0].split(':', 1)
                    if len(dataset_id_parts) == 2:
                        dataset, identifier = dataset_id_parts
                        resolution = parts[1]
                        
                        # Get counts for this combination
                        counts = {
                            "Strong Positive": 0,
                            "Weak Positive": 0,
                            "Weak Negative": 0,
                            "Negative": 0,
                            "Noise": 0
                        }
                        
                        query = """
                            SELECT labels, COUNT(*) 
                            FROM imag_with_seqs 
                            WHERE dataset = ? AND hic_path LIKE ? AND resolution = ? 
                            AND labels IS NOT NULL AND labels != ''
                            GROUP BY labels
                        """
                        source_cursor.execute(query, (dataset, f"%/{identifier}.hic", resolution))
                        
                        for label, count in source_cursor.fetchall():
                            if label in counts:
                                counts[label] = count
                        
                        button.update_counts(counts)
        finally:
            source_conn.close()


    
    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.load_current_grid()
    
    def next_page(self):
        """Go to next page"""
        self.current_page += 1
        self.load_current_grid()
        
    def load_current_grid(self):
        """Load current page of images"""
        try:
            if not self.data_manager.current_combination:
                self.status_label.setText("No combination selected")
                return
            
            # Get current combination info
            combo = self.data_manager.current_combination
            start_index = self.current_page * 25
            
            # Check if we need to expand
            while start_index >= len(combo.sample_indices) and start_index < combo.total_available:
                if not self.data_manager.expand_current_combination():
                    break
            
            # Load images for this page
            images = self.data_manager.get_grid_images(start_index)
            self.image_grid.load_images(images)
            
            # Update dataset info with label counts
            info_text = f"Combination: {combo.combination_key}\n"
            if images:
                first_metadata = images[0][1]
                info_text += f"Dataset: {first_metadata.get('dataset', 'N/A')}\n"
                info_text += f"HiC Path: {first_metadata.get('hic_path', 'N/A')}\n"
                info_text += f"Resolution: {first_metadata.get('resolution', 'N/A')}\n"
            
            # Add label counts - compact if small screen
            info_text += f"\nLabel Counts:\n"
            label_counts = self.get_current_combination_label_counts()
            
            if self.needs_scrolling:
                # Compact horizontal format for small screens
                count_parts = []
                for i, (label, count) in enumerate(label_counts.items(), 1):
                    count_parts.append(f"{i}:{count}")
                info_text += " | ".join(count_parts)
            else:
                # Normal vertical format
                for i, (label, count) in enumerate(label_counts.items(), 1):
                    info_text += f"  {i}. {label}: {count}\n"
            
            info_text += f"\nSamples: {len(combo.sample_indices)} loaded / {combo.total_available} available"
            self.dataset_info_label.setText(info_text)
            
            # Update page info
            total_pages = (len(combo.sample_indices) + 24) // 25
            self.page_label.setText(f"Page {self.current_page + 1} / {total_pages}")
            
            # Update navigation buttons
            self.prev_button.setEnabled(self.current_page > 0)
            self.next_button.setEnabled(True)
            
            # Update pending saves display
            self.update_pending_saves_display()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load images: {str(e)}")


    def get_current_combination_label_counts(self) -> dict:
        """Get label counts for current combination"""
        counts = {
            "Strong Positive": 0,
            "Weak Positive": 0,
            "Weak Negative": 0,
            "Negative": 0,
            "Noise": 0
        }
        
        if not self.data_manager.current_combination:
            return counts
        
        # Connect to source database and count labels
        source_conn = sqlite3.connect(self.data_manager.source_db_path)
        source_cursor = source_conn.cursor()
        
        try:
            # Get all key_ids for current combination
            key_ids = [kid for _, kid, _ in self.data_manager.current_combination.sample_indices]
            
            if key_ids:
                placeholders = ','.join(['?' for _ in key_ids])
                query = f"SELECT labels, COUNT(*) FROM imag_with_seqs WHERE key_id IN ({placeholders}) AND labels IS NOT NULL AND labels != '' GROUP BY labels"
                source_cursor.execute(query, key_ids)
                
                for label, count in source_cursor.fetchall():
                    if label in counts:
                        counts[label] = count
        finally:
            source_conn.close()
        
        return counts


    def write_labels(self):
        """Apply selected label to selected images"""
        selected_button = self.label_group.checkedButton()
        if not selected_button:
            QMessageBox.warning(self, "Warning", "Please select a label before applying.")
            return
        
        key_ids = self.image_grid.get_selected_key_ids()
        if not key_ids:
            QMessageBox.warning(self, "Warning", "Please select images to label.")
            return
        
        if selected_button == self.clear_radio:
            # Clear labels
            self.data_manager.clear_labels(key_ids)
            self.status_label.setText(f"Cleared labels for {len(key_ids)} images")
            # Update display to remove label numbers
            self.image_grid.update_label_numbers({kid: "" for kid in key_ids})
        else:
            # Apply label
            label = selected_button.text().split(") ", 1)[1] if ") " in selected_button.text() else selected_button.text()
            self.data_manager.update_labels(key_ids, label)
            self.status_label.setText(f"Applied '{label}' to {len(key_ids)} images")
            # Update display to show new label numbers immediately
            self.image_grid.update_label_numbers({kid: label for kid in key_ids})
        

        self.update_all_combination_counts()
        self.status_label.setStyleSheet("color: #A3BE8C; font-weight: bold; padding: 10px;")
        
        # Update displays
        self.update_pending_saves_display()
        
        # Update label counts in info
        self.update_dataset_info_display()
        
        # Auto-save if threshold reached
        if self.data_manager.get_pending_save_count() >= self.PENDING_COUNT:
            self.save_labels(auto_save=True)
    
    def update_dataset_info_display(self):
        """Update dataset info display without reloading images"""
        if not self.data_manager.current_combination:
            return
        
        combo = self.data_manager.current_combination
        
        info_text = f"Combination: {combo.combination_key}\n"
        
        # Get first image metadata for dataset info
        if self.image_grid.image_data:
            first_key_id, first_metadata = next(iter(self.image_grid.image_data.values()))
            info_text += f"Dataset: {first_metadata.get('dataset', 'N/A')}\n"
            info_text += f"HiC Path: {first_metadata.get('hic_path', 'N/A')}\n"
            info_text += f"Resolution: {first_metadata.get('resolution', 'N/A')}\n"
        
        # Add label counts
        info_text += f"\nLabel Counts:\n"
        label_counts = self.get_current_combination_label_counts()
        for i, (label, count) in enumerate(label_counts.items(), 1):
            info_text += f"  {i}. {label}: {count}\n"
        
        info_text += f"\nSamples: {len(combo.sample_indices)} loaded / {combo.total_available} available"
        self.dataset_info_label.setText(info_text)

    
    def save_labels(self, auto_save=False):
        """Save all pending labels to database"""
        pending_count = self.data_manager.get_pending_save_count()
        
        if pending_count == 0:
            if not auto_save:
                QMessageBox.information(self, "Info", "No pending labels to save.")
            return
        
        try:
            saved_count = self.data_manager.save_pending_labels()
            
            if auto_save:
                self.status_label.setText(f"Auto-saved {saved_count} labels")
            else:
                self.status_label.setText(f"Saved {saved_count} labels to database")
            
            self.status_label.setStyleSheet("color: #A3BE8C; font-weight: bold; padding: 10px;")
            
            # Update displays
            self.update_pending_saves_display()
            
            # UPDATE COMBINATION COUNTS AFTER SAVING
            self.update_all_combination_counts()
            
            # Also update the dataset info display to show new counts
            self.update_dataset_info_display()
            
            if not auto_save:
                QMessageBox.information(self, "Success", f"Successfully saved {saved_count} labels.")
            
        except Exception as e:
            self.status_label.setText(f"Error saving labels: {str(e)}")
            self.status_label.setStyleSheet("color: #BF616A; font-weight: bold; padding: 10px;")
            QMessageBox.critical(self, "Error", f"Failed to save labels: {str(e)}")
    
    def update_pending_saves_display(self):
        """Update the pending saves display"""
        pending_count = self.data_manager.get_pending_save_count()
        self.pending_saves_label.setText(f"Pending saves: {pending_count}")
        
        # Change color based on count
        if pending_count >= 100:
            self.pending_saves_label.setStyleSheet("color: #BF616A; font-weight: bold; padding: 10px;")
        elif pending_count >= 50:
            self.pending_saves_label.setStyleSheet("color: #EBCB8B; font-weight: bold; padding: 10px;")
        else:
            self.pending_saves_label.setStyleSheet("color: #D08770; font-weight: bold; padding: 10px;")
    
    def closeEvent(self, event):
        """Handle window close event"""
        pending_count = self.data_manager.get_pending_save_count()
        
        if pending_count > 0:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                f"You have {pending_count} unsaved labels.\n"
                "Do you want to save them before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                try:
                    self.save_labels()
                    event.accept()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save labels: {str(e)}")
                    event.ignore()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:  # Cancel
                event.ignore()
        else:
            event.accept()



class ImageLabelingApp:
    """Main application class"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("HiC Grid Image Labeling Tool")
    
    def run(self):
        """Run the application"""
        # Show splash screen
        self.splash = SplashScreen()
        
        # Connect signals
        self.splash.dataset_selected.connect(self.open_dataset)
        
        self.splash.show()
        
        # Run the application
        sys.exit(self.app.exec_())
    
    def open_dataset(self, source_db_id: int):
        """Open dataset for labeling"""
        try:
            data_manager = GridDataManager(source_db_id, self.splash.labeling_db_path)
            self.splash.hide()  # Hide instead of close
            self.show_labeling_window(data_manager)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to open dataset: {str(e)}")

    def show_labeling_window(self, data_manager: GridDataManager):
        """Show the main labeling window"""
        self.window = ImageLabelingWindow(data_manager, self.splash, 125)
        self.window.show()


if __name__ == "__main__":
    print("""
██╗      █████╗ ██████╗ ██╗ █████╗ ████████╗
██║     ██╔══██╗██╔══██╗██║██╔══██╗╚══██╔══╝
██║     ███████║██████╔╝██║███████║   ██║   
██║     ██╔══██║██╔══██╗██║██╔══██║   ██║   
███████╗██║  ██║██║  ██║██║██║  ██║   ██║   
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝   ╚═╝   

██╗███╗   ██╗████████╗███████╗██████╗ ███████╗██╗  ██╗
██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔════╝╚██╗██╔╝
██║██╔██╗ ██║   ██║   █████╗  ██████╔╝█████╗   ╚███╔╝ 
██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔══╝   ██╔██╗ 
██║██║ ╚████║   ██║   ███████╗██║  ██║██║     ██╔╝ ██╗
╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
v.0.0.13
Sean Moran
""")

    app = ImageLabelingApp()
    app.run()






