import os
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector, SpanSelector, Cursor
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from typing import Dict, List, Tuple, Optional, Union, Any
from pandastable import Table
from mpl_toolkits.mplot3d import Axes3D

# ==================== Custom Exceptions ====================
class AVOAnalyzerError(Exception):
    """Base exception for AVO Analyzer application"""
    pass

class DataValidationError(AVOAnalyzerError):
    """Exception for data validation errors"""
    pass

class FileOperationError(AVOAnalyzerError):
    """Exception for file operation errors"""
    pass

class ProcessingError(AVOAnalyzerError):
    """Exception for processing errors"""
    pass

# ==================== Project Management ====================
class ProjectManager:
    """Manages project data and settings"""
    
    def __init__(self):
        self.projects = {}
        self.current_project = None
        self.project_dir = os.path.join(os.path.expanduser("~"), "AVO_Analyzer_Projects")
        os.makedirs(self.project_dir, exist_ok=True)
    
    def create_project(self, name: str) -> bool:
        """Create a new project"""
        project_path = os.path.join(self.project_dir, name)
        if os.path.exists(project_path):
            messagebox.showerror("Error", f"Project '{name}' already exists")
            return False
        
        os.makedirs(project_path)
        self.projects[name] = {
            'path': project_path,
            'log_files': [],
            'zone_files': [],
            'settings': {},
            'created_at': pd.Timestamp.now().isoformat()
        }
        self.current_project = name
        self.save_project()
        return True
    
    def save_project(self) -> bool:
        """Save current project state"""
        if not self.current_project:
            return False
        
        project_path = self.projects[self.current_project]['path']
        config_path = os.path.join(project_path, 'project_config.json')
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.projects[self.current_project], f, indent=2)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save project: {str(e)}")
            return False
    
    def load_project(self, name: str) -> bool:
        """Load an existing project"""
        if name not in self.projects:
            return False
        
        project_path = self.projects[name]['path']
        config_path = os.path.join(project_path, 'project_config.json')
        
        if not os.path.exists(config_path):
            return False
        
        try:
            with open(config_path, 'r') as f:
                self.projects[name] = json.load(f)
            
            self.current_project = name
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load project: {str(e)}")
            return False
    
    def add_file_to_project(self, file_path: str, file_type: str) -> bool:
        """Add a file to the current project"""
        if not self.current_project:
            return False
        
        file_name = os.path.basename(file_path)
        project_path = self.projects[self.current_project]['path']
        dest_path = os.path.join(project_path, file_type, file_name)
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        try:
            import shutil
            shutil.copy2(file_path, dest_path)
            
            if file_type == 'logs':
                self.projects[self.current_project]['log_files'].append(file_name)
            elif file_type == 'zones':
                self.projects[self.current_project]['zone_files'].append(file_name)
            
            self.save_project()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add file to project: {str(e)}")
            return False

# ==================== Enhanced Data Processor ====================
class DataProcessor:
    """Handles all data processing operations including loading, validation, and fluid substitution"""
    
    def __init__(self):
        # Initialize data containers
        self.vp = None
        self.vs = None
        self.rho = None
        self.gr = None
        self.porosity = None
        self.sw = None
        self.vsh = None
        self.depth = None
        self.log_headers = None
        self.zone_tops = None
        self.blocked_data = None
        self.fluid_sub_zones = {}  # Store fluid substitution zones
        self.elastic_moduli = {}   # Store calculated elastic moduli
        
    def load_log_data(self, file_path: str) -> Tuple[bool, str]:
        """Load well log data from ASCII text file with validation
        
        Args:
            file_path: Path to the well log file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not os.path.exists(file_path):
            raise FileOperationError(f"File not found: {file_path}")
        
        try:
            # Use pandas for faster loading of large files
            df = pd.read_csv(file_path, delim_whitespace=True, header=0)
            
            self.log_headers = df.columns.tolist()
            
            required_columns = ['Depth', 'Vp', 'Vs', 'Rho', 'GR', 'Phi', 'Sw', 'Vsh']
            missing_cols = [col for col in required_columns if col not in self.log_headers]
            
            if missing_cols:
                return False, f"Required columns not found: {', '.join(missing_cols)}"
            
            # Load data with type conversion
            try:
                self.depth = df['Depth'].values
                self.vp = df['Vp'].values
                self.vs = df['Vs'].values
                self.rho = df['Rho'].values
                self.gr = df['GR'].values
                self.porosity = df['Phi'].values
                self.sw = df['Sw'].values
                self.vsh = df['Vsh'].values
            except ValueError as e:
                return False, f"Failed to convert data to numeric values: {str(e)}"
            
            # Validate loaded data
            is_valid, message = self.validate_log_data()
            if not is_valid:
                return False, message
                
            # Calculate elastic moduli
            self.elastic_moduli = self.calculate_elastic_moduli()
                
            return True, f"Successfully loaded {len(self.depth)} data points"
            
        except Exception as e:
            return False, f"Failed to load well log file: {str(e)}"
    
    def validate_log_data(self) -> Tuple[bool, str]:
        """Validate loaded log data for consistency and completeness
        
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        if self.depth is None:
            return False, "No depth data loaded"
            
        # Check for consistent array lengths
        arrays = [self.vp, self.vs, self.rho, self.gr, self.porosity, self.sw, self.vsh]
        names = ['Vp', 'Vs', 'Rho', 'GR', 'Phi', 'Sw', 'Vsh']
        
        for arr, name in zip(arrays, names):
            if arr is None:
                return False, f"Missing {name} data"
            if len(arr) != len(self.depth):
                return False, f"Inconsistent length for {name} data"
                
        # Check for NaN values
        for arr, name in zip(arrays, names):
            nan_count = np.isnan(arr).sum()
            if nan_count > len(arr) * 0.5:  # More than 50% NaN
                return False, f"Too many NaN values in {name} data ({nan_count}/{len(arr)})"
                
        # Check for physically reasonable values
        if np.any(self.vp <= 0) or np.any(self.vs < 0) or np.any(self.rho <= 0):
            return False, "Invalid values in Vp, Vs, or Rho (must be positive)"
            
        if np.any(self.porosity < 0) or np.any(self.porosity > 0.5):
            return False, "Porosity values outside reasonable range (0-0.5)"
            
        if np.any(self.sw < 0) or np.any(self.sw > 1):
            return False, "Water saturation values outside valid range (0-1)"
            
        return True, "Data validation passed"
    
    def calculate_elastic_moduli(self) -> Dict[str, np.ndarray]:
        """Calculate elastic moduli from well log data"""
        if self.vp is None or self.vs is None or self.rho is None:
            return {}
        
        # Convert density from g/cc to kg/m3
        rho_kgm3 = self.rho * 1000
        
        # Calculate bulk modulus
        k = rho_kgm3 * (self.vp**2 - 4/3 * self.vs**2)
        
        # Calculate shear modulus
        mu = rho_kgm3 * self.vs**2
        
        # Calculate Young's modulus
        with np.errstate(divide='ignore', invalid='ignore'):
            E = np.where((3 * k + mu) != 0, 9 * k * mu / (3 * k + mu), np.nan)
        
        # Calculate Poisson's ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            pr = np.where((3 * k + mu) != 0, (3 * k - 2 * mu) / (2 * (3 * k + mu)), np.nan)
        
        # Calculate lambda parameter
        lam = k - 2/3 * mu
        
        # Calculate P-wave and S-wave impedances
        ip = self.vp * rho_kgm3
        is_ = self.vs * rho_kgm3
        
        return {
            'Bulk Modulus': k,
            'Shear Modulus': mu,
            'Youngs Modulus': E,
            'Poissons Ratio': pr,
            'Lambda': lam,
            'P-Impedance': ip,
            'S-Impedance': is_
        }
    
    def load_zone_tops(self, file_path: str) -> Tuple[bool, str]:
        """Load zone tops data from ASCII text file
        
        Args:
            file_path: Path to the zone tops file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.depth is None:
            return False, "Please load well logs first."
            
        try:
            data = np.genfromtxt(file_path, delimiter=None, dtype=None, names=True, 
                                encoding=None, autostrip=True)
            
            if 'Depth' not in data.dtype.names or 'Tops' not in data.dtype.names:
                return False, "Required columns 'Depth' and 'Tops' not found"
            
            self.zone_tops = np.array([(float(d), str(t)) for d, t in data[['Depth', 'Tops']]],
                                      dtype=[('Depth', float), ('Tops', 'U100')])
            
            self.zone_tops.sort(order='Depth')
            self.calculate_blocked_data()
            
            return True, f"Successfully loaded {len(self.zone_tops)} zone tops"
            
        except Exception as e:
            return False, f"Failed to load zone tops: {str(e)}"
    
    def calculate_blocked_data(self) -> None:
        """Calculate average values for each zone between tops"""
        if self.zone_tops is None or self.depth is None or len(self.zone_tops) == 0:
            self.blocked_data = None
            return
            
        zone_names = self.zone_tops['Tops']
        zone_depths = self.zone_tops['Depth']
        num_zones = len(zone_names)
        
        temp_blocked_data = {
            'Zone': [], 'Depth': [], 'Vp': [], 'Vs': [], 'Rho': [],
            'GR': [], 'Phi': [], 'Sw': [], 'Vsh': [], 'Thickness': []
        }
        
        for i in range(num_zones):
            top_depth = zone_depths[i]
            
            if i + 1 < num_zones:
                bottom_depth = zone_depths[i + 1]
            else:
                bottom_depth = np.max(self.depth) 
                if top_depth >= bottom_depth:
                    print(f"Warning: Last zone top '{zone_names[i]}' at {top_depth}m is beyond or at end of log data at {bottom_depth}m. Skipping.")
                    continue
            
            mask = (self.depth >= top_depth) & (self.depth <= bottom_depth)
            if np.any(mask):
                zone_mid_depth = (top_depth + bottom_depth) / 2
                
                temp_blocked_data['Zone'].append(zone_names[i])
                temp_blocked_data['Depth'].append(zone_mid_depth)
                
                temp_blocked_data['Vp'].append(np.nanmean(self.vp[mask]))
                temp_blocked_data['Vs'].append(np.nanmean(self.vs[mask]))
                temp_blocked_data['Rho'].append(np.nanmean(self.rho[mask]))
                temp_blocked_data['GR'].append(np.nanmean(self.gr[mask]))
                temp_blocked_data['Phi'].append(np.nanmean(self.porosity[mask]))
                temp_blocked_data['Sw'].append(np.nanmean(self.sw[mask]))
                temp_blocked_data['Vsh'].append(np.nanmean(self.vsh[mask]))
                
                # Calculate thickness
                if i < num_zones - 1:
                    thickness = zone_depths[i+1] - zone_depths[i]
                else:
                    thickness = np.max(self.depth) - zone_depths[i]
                temp_blocked_data['Thickness'].append(thickness)
                
            else:
                print(f"Warning: No log data found for zone '{zone_names[i]}' between depth {top_depth} and {bottom_depth}. This zone will be skipped in blocked data.")
                
        df_blocked = pd.DataFrame(temp_blocked_data).dropna()
        
        if df_blocked.empty:
            self.blocked_data = None
        else:
            self.blocked_data = df_blocked.to_dict(orient='list')
    
    def apply_fluid_substitution(self, zone_name: str, fluid_type: str, 
                                vp_fluid: float, vs_fluid: float, rho_fluid: float,
                                method: str = "Gassmann") -> Tuple[bool, str]:
        """Apply fluid substitution to the selected zone
        
        Args:
            zone_name: Name of the zone to apply fluid substitution
            fluid_type: Type of fluid (Brine, Oil, Gas)
            vp_fluid: P-wave velocity of the fluid (m/s)
            vs_fluid: S-wave velocity of the fluid (m/s)
            rho_fluid: Density of the fluid (g/cc)
            method: Fluid substitution method ("Gassmann" or "Xu-Payne")
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.blocked_data is None or not self.blocked_data['Zone']:
            return False, "No blocked data available"
            
        try:
            zone_idx = self.blocked_data['Zone'].index(zone_name)
        except ValueError:
            return False, f"Zone '{zone_name}' not found in blocked data"
            
        # Get original zone properties
        vp_orig = self.blocked_data['Vp'][zone_idx]
        vs_orig = self.blocked_data['Vs'][zone_idx]
        rho_orig = self.blocked_data['Rho'][zone_idx]
        phi = self.blocked_data['Phi'][zone_idx]
        
        # Check for valid values
        if np.isnan([vp_orig, vs_orig, rho_orig, phi]).any():
            return False, f"Zone '{zone_name}' has missing values for required properties"
            
        # Apply fluid substitution
        try:
            if method == "Gassmann":
                vp_new, vs_new, rho_new = self.gassmann_fluid_substitution(
                    vp_orig, vs_orig, rho_orig, phi, 
                    vp_fluid, vs_fluid, rho_fluid
                )
            elif method == "Xu-Payne":
                vp_new, vs_new, rho_new = self.xu_payne_fluid_substitution(
                    vp_orig, vs_orig, rho_orig, phi, 
                    vp_fluid, vs_fluid, rho_fluid
                )
            else:
                return False, f"Unknown fluid substitution method: {method}"
            
            # Store fluid substitution results
            fluid_case_name = f"{zone_name}_{fluid_type}"
            self.fluid_sub_zones[fluid_case_name] = {
                'Vp': vp_new,
                'Vs': vs_new,
                'Rho': rho_new,
                'Zone': zone_name,
                'FluidType': fluid_type,
                'Method': method
            }
            
            # Update blocked data with fluid-substituted values
            self.blocked_data['Vp'][zone_idx] = vp_new
            self.blocked_data['Vs'][zone_idx] = vs_new
            self.blocked_data['Rho'][zone_idx] = rho_new
            
            return True, (f"Fluid substitution applied to {zone_name} using {method} method.\n"
                         f"Vp: {vp_orig:.2f} → {vp_new:.2f} m/s\n"
                         f"Vs: {vs_orig:.2f} → {vs_new:.2f} m/s\n"
                         f"Rho: {rho_orig:.2f} → {rho_new:.2f} g/cc")
                           
        except Exception as e:
            return False, f"Fluid substitution failed: {str(e)}"
    
    def gassmann_fluid_substitution(self, vp_orig: float, vs_orig: float, rho_orig: float, 
                                   phi: float, vp_fluid: float, vs_fluid: float, 
                                   rho_fluid: float) -> Tuple[float, float, float]:
        """Apply Gassmann fluid substitution to calculate new elastic properties
        
        Args:
            vp_orig: Original P-wave velocity (m/s)
            vs_orig: Original S-wave velocity (m/s)
            rho_orig: Original bulk density (g/cc)
            phi: Porosity (fraction)
            vp_fluid: P-wave velocity of new fluid (m/s)
            vs_fluid: S-wave velocity of new fluid (m/s)
            rho_fluid: Density of new fluid (g/cc)
            
        Returns:
            Tuple of (vp_new, vs_new, rho_new) - New elastic properties
        """
        # Input validation
        if vp_orig <= 0 or vs_orig <= 0 or rho_orig <= 0:
            raise ValueError("Original velocities and density must be positive")
        
        if phi < 0 or phi > 1:
            raise ValueError("Porosity must be between 0 and 1")
        
        # Convert units as needed
        rho_orig_kgm3 = rho_orig * 1000  # g/cc to kg/m3
        rho_fluid_kgm3 = rho_fluid * 1000  # g/cc to kg/m3
        
        # Calculate original moduli
        k_orig = rho_orig_kgm3 * (vp_orig**2 - 4/3 * vs_orig**2)  # Bulk modulus
        mu_orig = rho_orig_kgm3 * vs_orig**2  # Shear modulus
        
        # Calculate mineral properties
        rho_mineral = (rho_orig_kgm3 - phi * rho_fluid_kgm3) / (1 - phi)
        
        # Estimate mineral bulk modulus (using simplified approach)
        k_mineral = k_orig * (1 + phi) / (1 - phi)  # Simplified estimation
        
        # Calculate fluid bulk modulus
        k_fluid = rho_fluid_kgm3 * vp_fluid**2  # For fluids, Vs = 0
        
        # Apply Gassmann's equation
        k_sat = k_orig
        k_dry = (k_sat * (k_mineral * (1 - phi) + k_fluid * phi) - k_mineral * k_fluid * phi) / \
                (k_mineral * (1 - phi) + k_fluid * phi - k_sat * phi)
        
        k_new = k_dry + (1 - k_dry/k_mineral)**2 / (phi/k_fluid + (1-phi)/k_mineral - k_dry/k_mineral**2)
        
        # Shear modulus remains unchanged (Gassmann assumption)
        mu_new = mu_orig
        
        # Calculate new density
        rho_new_kgm3 = rho_mineral * (1 - phi) + rho_fluid_kgm3 * phi
        
        # Calculate new velocities
        vp_new = np.sqrt((k_new + 4/3 * mu_new) / rho_new_kgm3)
        vs_new = np.sqrt(mu_new / rho_new_kgm3)
        
        # Convert density back to g/cc
        rho_new = rho_new_kgm3 / 1000
        
        return vp_new, vs_new, rho_new
    
    def xu_payne_fluid_substitution(self, vp_orig: float, vs_orig: float, rho_orig: float, 
                                   phi: float, vp_fluid: float, vs_fluid: float, 
                                   rho_fluid: float, k_mineral: float = 36.6, 
                                   mu_mineral: float = 44.0) -> Tuple[float, float, float]:
        """Apply Xu-Payne fluid substitution for more accurate results
        
        Args:
            vp_orig, vs_orig, rho_orig: Original properties
            phi: Porosity
            vp_fluid, vs_fluid, rho_fluid: New fluid properties
            k_mineral: Mineral bulk modulus (GPa)
            mu_mineral: Mineral shear modulus (GPa)
            
        Returns:
            Tuple of (vp_new, vs_new, rho_new)
        """
        # Input validation
        if vp_orig <= 0 or vs_orig <= 0 or rho_orig <= 0:
            raise ValueError("Original velocities and density must be positive")
        
        if phi < 0 or phi > 1:
            raise ValueError("Porosity must be between 0 and 1")
        
        # Convert units
        rho_orig_kgm3 = rho_orig * 1000
        rho_fluid_kgm3 = rho_fluid * 1000
        
        # Calculate original moduli
        k_orig = rho_orig_kgm3 * (vp_orig**2 - 4/3 * vs_orig**2) / 1e9  # Convert to GPa
        mu_orig = rho_orig_kgm3 * vs_orig**2 / 1e9  # Convert to GPa
        
        # Calculate fluid modulus
        k_fluid = rho_fluid_kgm3 * vp_fluid**2 / 1e9  # Convert to GPa
        
        # Calculate dry rock frame moduli using Gassmann
        k_dry = (k_orig * (k_mineral * (1 - phi) + k_fluid * phi) - k_mineral * k_fluid * phi) / \
                (k_mineral * (1 - phi) + k_fluid * phi - k_orig * phi)
        
        # Apply Xu-Payne correction for shear modulus
        mu_dry = mu_orig * (1 - phi) ** (1 + 1.5 * phi)  # Empirical relationship
        
        # Apply Gassmann for new fluid
        k_new = k_dry + (1 - k_dry/k_mineral)**2 / (phi/k_fluid + (1-phi)/k_mineral - k_dry/k_mineral**2)
        mu_new = mu_dry  # Shear modulus unchanged
        
        # Calculate new density
        rho_new_kgm3 = (k_mineral * 1000 * (1 - phi) + rho_fluid_kgm3 * phi)
        
        # Calculate new velocities
        vp_new = np.sqrt((k_new * 1e9 + 4/3 * mu_new * 1e9) / rho_new_kgm3)
        vs_new = np.sqrt(mu_new * 1e9 / rho_new_kgm3)
        
        # Convert density back to g/cc
        rho_new = rho_new_kgm3 / 1000
        
        return vp_new, vs_new, rho_new
    
    def get_zone_data(self, zone_name: str) -> Optional[Dict[str, float]]:
        """Get the data for a specific zone
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            Dictionary with zone data or None if not found
        """
        if self.blocked_data is None or zone_name not in self.blocked_data['Zone']:
            return None
            
        try:
            zone_idx = self.blocked_data['Zone'].index(zone_name)
            return {
            'Zone': zone_name,
            'Vp': self.blocked_data['Vp'][zone_idx],
            'Vs': self.blocked_data['Vs'][zone_idx],
            'Rho': self.blocked_data['Rho'][zone_idx],
            'Depth': self.blocked_data['Depth'][zone_idx],
            'GR': self.blocked_data['GR'][zone_idx],
            'Phi': self.blocked_data['Phi'][zone_idx],
            'Sw': self.blocked_data['Sw'][zone_idx],
            'Vsh': self.blocked_data['Vsh'][zone_idx],
            'Thickness': self.blocked_data['Thickness'][zone_idx],
            'Lithology': self.blocked_data['Lithology'][zone_idx] if 'Lithology' in self.blocked_data else None
            }
   
        except (ValueError, IndexError):
            return None
    
    def export_blocked_data(self, file_path: str) -> Tuple[bool, str]:
        """Export blocked data to CSV file
        
        Args:
            file_path: Path to save the CSV file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.blocked_data is None:
            return False, "No data available to export"
            
        try:
            df = pd.DataFrame(self.blocked_data)
            df.to_csv(file_path, index=False)
            return True, f"Data exported to {file_path}"
        except Exception as e:
            return False, f"Failed to export data: {str(e)}"
    
    def export_to_excel(self, file_path: str) -> Tuple[bool, str]:
        """Export blocked data to Excel with multiple sheets"""
        if self.blocked_data is None:
            return False, "No data available to export"
        
        try:
            with pd.ExcelWriter(file_path) as writer:
                # Export blocked data
                df_blocked = pd.DataFrame(self.blocked_data)
                df_blocked.to_excel(writer, sheet_name='Blocked Data', index=False)
                
                # Export fluid substitution results
                if self.fluid_sub_zones:
                    df_fluid = pd.DataFrame(self.fluid_sub_zones).T
                    df_fluid.to_excel(writer, sheet_name='Fluid Substitution', index=False)
                
                # Export original zone tops
                if self.zone_tops is not None:
                    df_tops = pd.DataFrame(self.zone_tops)
                    df_tops.to_excel(writer, sheet_name='Zone Tops', index=False)
                
                # Export elastic moduli if available
                if self.elastic_moduli:
                    df_moduli = pd.DataFrame(self.elastic_moduli)
                    df_moduli.to_excel(writer, sheet_name='Elastic Moduli', index=False)
            
            return True, f"Data exported to Excel file: {file_path}"
        
        except Exception as e:
            return False, f"Failed to export to Excel: {str(e)}"

# ==================== Enhanced Visualization Manager ====================
class VisualizationManager:
    """Handles all visualization operations including plots and exports"""
    
    def __init__(self, data_processor: DataProcessor, master=None):
        self.data_processor = data_processor
        self.master = master  # Store reference to the master window
        self.fig = None
        self.axes = None
        self.canvas = None
        self.synthetic_fig = None
        self.synthetic_canvas = None
        self.crossplot_fig = None
        self.crossplot_canvas = None
        
    def initialize_log_plots(self, parent_frame) -> None:
        """Initialize log visualization area with empty plots
        
        Args:
            parent_frame: Tkinter frame to place the plots
        """
        self.fig, self.axes = plt.subplots(nrows=1, ncols=7, figsize=(15, 8), sharey=True)
        self.fig.subplots_adjust(wspace=0.05)
        
        default_headers = ['Vp', 'Vs', 'Rho', 'GR', 'Phi', 'Sw', 'Vsh']
        
        for i, ax in enumerate(self.axes):
            ax.set_ylabel('Depth (m)' if i == 0 else '', fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_title(default_headers[i], fontsize=11)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim(0, 1000)
            ax.set_xlim(0, 1)
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, parent_frame)
        toolbar.update()
        
        # Add interactive tools
        self.add_interactive_tools(self.canvas, self.fig)
        self.add_cursor_crosshair(self.canvas, self.fig)
    
    def add_interactive_tools(self, canvas, fig):
        """Add interactive tools to plots"""
        # Add rectangle selector for zooming
        def on_select(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            
            # Set new axis limits
            for ax in fig.axes:
                ax.set_xlim(min(x1, x2), max(x1, x2))
                ax.set_ylim(min(y1, y2), max(y1, y2))
            
            canvas.draw()
        
        rect_selector = RectangleSelector(
            fig.axes[0], on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        # Add span selector for depth range selection
        def on_depth_select(min_depth, max_depth):
            # Highlight selected depth range
            for ax in fig.axes:
                ax.axhspan(min_depth, max_depth, alpha=0.2, color='yellow')
            
            canvas.draw()
        
        span_selector = SpanSelector(
            fig.axes[0], on_depth_select,
            'vertical', useblit=True,
            button=[1],  # Left mouse button
            minspan=5
        )
        
        return rect_selector, span_selector
    
    def add_cursor_crosshair(self, canvas, fig):
        """Add crosshair cursor that follows mouse movement"""
        cursor = Cursor(fig.axes[0], useblit=True, color='red', linewidth=1)
        
        # Add data display
        def on_motion(event):
            if event.inaxes == fig.axes[0]:
                # Find closest data point
                depth = event.ydata
                if depth is not None and self.data_processor.depth is not None:
                    idx = np.abs(self.data_processor.depth - depth).argmin()
                    
                    # Update status text with data values
                    status_text = (
                        f"Depth: {self.data_processor.depth[idx]:.2f}m, "
                        f"Vp: {self.data_processor.vp[idx]:.2f}m/s, "
                        f"Vs: {self.data_processor.vs[idx]:.2f}m/s, "
                        f"Rho: {self.data_processor.rho[idx]:.2f}g/cc"
                    )
                    
                    # Update status bar if available
                    if hasattr(canvas, 'get_tk_widget'):
                        master = canvas.get_tk_widget().master
                        if hasattr(master, 'status_bar'):
                            master.status_bar.config(text=status_text)
        
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        
        return cursor
        
    def update_log_plots(self) -> None:
        """Update log visualization with loaded data and zone tops"""
        if self.fig is None or self.axes is None:
            return
            
        for ax in self.axes:
            ax.clear()
        
        # Check if we need to reset to original plots
        if self.master is not None and hasattr(self.master, 'original_plot_state') and not self.master.original_plot_state:
            # Just update with original data, no averages
            pass
        
        log_curves = [
            (self.data_processor.vp, 'Vp'), 
            (self.data_processor.vs, 'Vs'), 
            (self.data_processor.rho, 'Rho'), 
            (self.data_processor.gr, 'GR'), 
            (self.data_processor.porosity, 'Phi'), 
            (self.data_processor.sw, 'Sw'), 
            (self.data_processor.vsh, 'Vsh')
        ]
        
        if self.data_processor.depth is not None and len(self.data_processor.depth) > 0:
            for i, (data, label) in enumerate(log_curves):
                self._plot_log_track(self.axes[i], self.data_processor.depth, data, 
                                    'Depth (m)' if i == 0 else '', label)
        else:
            self.initialize_log_plots(self.canvas.get_tk_widget().master)
            
        if self.data_processor.zone_tops is not None:
            self._plot_zone_tops()
        
        if self.data_processor.blocked_data and self.data_processor.blocked_data['Zone']:
            self._plot_blocked_data()
        
        self.canvas.draw()

    def plot_blocked_averages(self) -> None:
        """Plot the average values from blocked data as constant lines on the log plots"""
        if self.data_processor.blocked_data is None or not self.data_processor.blocked_data['Zone']:
            return
        
        # Get zone tops data
        if self.data_processor.zone_tops is None:
            return
        
        # List of parameters in the order of the axes
        params = ['Vp', 'Vs', 'Rho', 'GR', 'Phi', 'Sw', 'Vsh']
        
        # Get the blocked data
        blocked_data = self.data_processor.blocked_data
        
        # Get zone tops
        zone_tops = self.data_processor.zone_tops
        zone_names = zone_tops['Tops']
        zone_depths = zone_tops['Depth']
        num_zones = len(zone_names)
        
        # For each parameter, plot the average values as constant lines
        for i, param in enumerate(params):
            ax = self.axes[i]
            
            # For each zone, plot a constant line
            for j in range(num_zones):
                zone_name = blocked_data['Zone'][j]
                if zone_name not in zone_names:
                    continue
                
                # Get the zone index in zone_tops
                zone_idx = np.where(zone_names == zone_name)[0]
                if len(zone_idx) == 0:
                    continue
                zone_idx = zone_idx[0]
                
                # Get the top and bottom depths for this zone
                top_depth = zone_depths[zone_idx]
                
                if zone_idx < num_zones - 1:
                    bottom_depth = zone_depths[zone_idx + 1]
                else:
                    bottom_depth = np.max(self.data_processor.depth)
                
                # Get the average value for this parameter
                avg_value = blocked_data[param][j]
                
                # Plot a constant line for the average value
                #ax.axhline(y=bottom_depth, xmin=0, xmax=1, color='red', linestyle='-', alpha=1, linewidth=1)
                #ax.axhline(y=top_depth, xmin=0, xmax=1, color='red', linestyle='-', alpha=1, linewidth=1)
                #ax.axvline(x=avg_value, ymin=top_depth/np.max(self.data_processor.depth), 
                          #ymax=bottom_depth/np.max(self.data_processor.depth), 
                          #color='red', linestyle='-', alpha=1, linewidth=4)
                
                # Fill the area to create a horizontal band
                ax.fill_betweenx([top_depth, bottom_depth], avg_value, avg_value, 
                                 color='red', alpha=0.7, linewidth=2)
        
        # Redraw the canvas
        if self.canvas:
            self.canvas.draw()
    
    def _plot_log_track(self, ax, depth, data, ylabel, xlabel) -> None:
        """Helper function to plot a single log track"""
        valid_indices = ~np.isnan(data) & ~np.isnan(depth)
        plot_depth = depth[valid_indices]
        plot_data = data[valid_indices]
        
        if len(plot_data) > 0:
            ax.plot(plot_data, plot_depth, 'b-', linewidth=0.5, label='Original')
            
            min_depth = np.min(plot_depth)
            max_depth = np.max(plot_depth)
            ax.set_ylim(max_depth, min_depth)
            
            data_min = np.nanmin(plot_data)
            data_max = np.nanmax(plot_data)
            if data_min == data_max:
                ax.set_xlim(data_min - 0.5, data_max + 0.5)
            else:
                data_range = data_max - data_min
                ax.set_xlim(data_min - 0.1 * data_range, data_max + 0.1 * data_range)
        else:
            ax.set_ylim(self.data_processor.depth.max() if self.data_processor.depth is not None and len(self.data_processor.depth) > 0 else 1000, 
                        self.data_processor.depth.min() if self.data_processor.depth is not None and len(self.data_processor.depth) > 0 else 0)
            ax.set_xlim(0, 1)
            
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_title(xlabel, fontsize=11)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    def _plot_zone_tops(self) -> None:
        """Plot zone tops as horizontal lines"""
        if self.data_processor.zone_tops is None or self.axes is None:
            return
            
        for i, ax in enumerate(self.axes):
            for j, row in enumerate(self.data_processor.zone_tops):
                depth = row['Depth']
                zone_name = row['Tops']
                
                ax.axhline(y=depth, color='r', linestyle='--', linewidth=0.2)
                
                if i == 0:
                    y_offset = (ax.get_ylim()[0] - ax.get_ylim()[1]) * 0.01
                    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
                            depth + y_offset, 
                            zone_name, 
                            verticalalignment='bottom', 
                            fontsize=8)
    
    def _plot_blocked_data(self) -> None:
        """Plot blocked data as colored blocks"""
        if (self.data_processor.blocked_data is None or 
            not self.data_processor.blocked_data['Zone'] or 
            self.axes is None):
            return
            
        original_zone_depths = self.data_processor.zone_tops['Depth']
        num_blocked_zones = len(self.data_processor.blocked_data['Zone'])
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_blocked_zones))
        
        for i in range(num_blocked_zones):
            zone_name = self.data_processor.blocked_data['Zone'][i]
            
            original_zone_idx = next((j for j, name in enumerate(self.data_processor.zone_tops['Tops']) if name == zone_name), None)
            if original_zone_idx is not None:
                top_depth = original_zone_depths[original_zone_idx]
                
                if original_zone_idx + 1 < len(original_zone_depths):
                    bottom_depth = original_zone_depths[original_zone_idx + 1]
                else:
                    bottom_depth = np.max(self.data_processor.depth)
                    
                mask = (self.data_processor.depth >= top_depth) & (self.data_processor.depth <= bottom_depth)
                
                if np.any(mask):
                    for j, ax in enumerate(self.axes):
                        ax.fill_betweenx(
                            self.data_processor.depth[mask], 
                            ax.get_xlim()[0], 
                            ax.get_xlim()[1],
                            color=colors[i],
                            alpha=0.2
                        )
    
    def create_crossplot(self, parent_frame, x_param: str, y_param: str, 
                       color_param: Optional[str] = None) -> None:
        """Create a crossplot of two parameters with optional color coding"""
        # Clear previous plots
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        self.crossplot_fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get data arrays
        x_data = None
        y_data = None
        
        # Check if parameter is in elastic moduli
        if self.master is not None and hasattr(self.master, 'elastic_moduli_data') and self.master.elastic_moduli_data:
            if x_param in self.master.elastic_moduli_data:
                x_data = np.array(self.master.elastic_moduli_data[x_param])
            if y_param in self.master.elastic_moduli_data:
                y_data = np.array(self.master.elastic_moduli_data[y_param])
        
        # If not found in elastic moduli, get from data_processor
        if x_data is None:
            # Handle special case for P-Impedance and S-Impedance
            if x_param == "P-Impedance":
                x_data = self.data_processor.elastic_moduli.get('P-Impedance', None)
            elif x_param == "S-Impedance":
                x_data = self.data_processor.elastic_moduli.get('S-Impedance', None)
            else:
                x_data = getattr(self.data_processor, x_param.lower(), None)
        
        if y_data is None:
            # Handle special case for P-Impedance and S-Impedance
            if y_param == "P-Impedance":
                y_data = self.data_processor.elastic_moduli.get('P-Impedance', None)
            elif y_param == "S-Impedance":
                y_data = self.data_processor.elastic_moduli.get('S-Impedance', None)
            else:
                y_data = getattr(self.data_processor, y_param.lower(), None)
        
        if x_data is None or y_data is None:
            ax.text(0.5, 0.5, "Invalid parameters", ha='center', va='center')
        else:
            if color_param:
                color_data = getattr(self.data_processor, color_param.lower(), None)
                scatter = ax.scatter(x_data, y_data, c=color_data, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=ax, label=color_param)
            else:
                ax.scatter(x_data, y_data, alpha=0.6)
            
            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            ax.set_title(f"{x_param} vs {y_param}")
            ax.grid(True, alpha=0.3)
        
        self.crossplot_canvas = FigureCanvasTkAgg(self.crossplot_fig, master=parent_frame)
        self.crossplot_canvas.draw()
        self.crossplot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.crossplot_canvas, parent_frame)
        toolbar.update()
    
    def create_3d_visualization(self, parent_frame, x_param: str, y_param: str, z_param: str) -> None:
        """Create a 3D visualization of three parameters"""
        # Clear previous plots
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        self.plot3d_fig = plt.figure(figsize=(10, 8))
        ax = self.plot3d_fig.add_subplot(111, projection='3d')
        
        # Get data arrays
        x_data = None
        y_data = None
        z_data = None
        
        # Check if parameter is in elastic moduli
        if self.master is not None and hasattr(self.master, 'elastic_moduli_data') and self.master.elastic_moduli_data:
            if x_param in self.master.elastic_moduli_data:
                x_data = np.array(self.master.elastic_moduli_data[x_param])
            if y_param in self.master.elastic_moduli_data:
                y_data = np.array(self.master.elastic_moduli_data[y_param])
            if z_param in self.master.elastic_moduli_data:
                z_data = np.array(self.master.elastic_moduli_data[z_param])
        
        # If not found in elastic moduli, get from data_processor
        if x_data is None:
            # Handle special case for P-Impedance and S-Impedance
            if x_param == "P-Impedance":
                x_data = self.data_processor.elastic_moduli.get('P-Impedance', None)
            elif x_param == "S-Impedance":
                x_data = self.data_processor.elastic_moduli.get('S-Impedance', None)
            else:
                x_data = getattr(self.data_processor, x_param.lower(), None)
        
        if y_data is None:
            # Handle special case for P-Impedance and S-Impedance
            if y_param == "P-Impedance":
                y_data = self.data_processor.elastic_moduli.get('P-Impedance', None)
            elif y_param == "S-Impedance":
                y_data = self.data_processor.elastic_moduli.get('S-Impedance', None)
            else:
                y_data = getattr(self.data_processor, y_param.lower(), None)
        
        if z_data is None:
            # Handle special case for P-Impedance and S-Impedance
            if z_param == "P-Impedance":
                z_data = self.data_processor.elastic_moduli.get('P-Impedance', None)
            elif z_param == "S-Impedance":
                z_data = self.data_processor.elastic_moduli.get('S-Impedance', None)
            else:
                z_data = getattr(self.data_processor, z_param.lower(), None)
        
        if x_data is None or y_data is None or z_data is None:
            ax.text2D(0.5, 0.5, "Invalid parameters", ha='center', va='center')
        else:
            ax.scatter(x_data, y_data, z_data, c='b', marker='o', alpha=0.5)
            
            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            ax.set_zlabel(z_param)
            ax.set_title(f"3D Plot: {x_param}, {y_param}, {z_param}")
        
        self.plot3d_canvas = FigureCanvasTkAgg(self.plot3d_fig, master=parent_frame)
        self.plot3d_canvas.draw()
        self.plot3d_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.plot3d_canvas, parent_frame)
        toolbar.update()
    
    def create_synthetic_seismogram(self, parent_frame, top_zone_1: Dict[str, float], 
                                  base_zone_1: Dict[str, float], top_zone_2: Dict[str, float], 
                                  base_zone_2: Dict[str, float], freq: int, 
                                  angles: np.ndarray, avo_method: str = "Aki-Richards",
                                  is_substituted: bool = False, 
                                  original_rc_2: Optional[np.ndarray] = None) -> None:
        """Create synthetic seismogram visualization with three columns
        
        Args:
            parent_frame: Tkinter frame to place the plots
            top_zone_1: Data for top zone of dataset 1
            base_zone_1: Data for base zone of dataset 1
            top_zone_2: Data for top zone of dataset 2
            base_zone_2: Data for base zone of dataset 2
            freq: Ricker wavelet frequency
            angles: Array of angles for AVO analysis
            avo_method: AVO approximation method to use
            is_substituted: Flag indicating if fluid substitution has been applied
            original_rc_2: Original reflection coefficients for dataset 2 (before substitution)
        """
        # Clear previous plots
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        # Create figure for synthetic seismogram with 3 subplots
        self.synthetic_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
        self.synthetic_fig.subplots_adjust(wspace=0.3, bottom=0.15)
        
        wavelet_duration = 0.1
        dt_wavelet = 0.001
        wavelet = self.generate_ricker_wavelet(freq, duration=wavelet_duration, dt=dt_wavelet)
        
        # Calculate reflection coefficients for Dataset 1
        if avo_method == "Aki-Richards":
            rc_1 = self.aki_richards(
                top_zone_1['Vp'], top_zone_1['Vs'], top_zone_1['Rho'],
                base_zone_1['Vp'], base_zone_1['Vs'], base_zone_1['Rho'],
                angles
            )
        elif avo_method == "Shuey":
            rc_1 = self.shuey_approximation(
                top_zone_1['Vp'], top_zone_1['Vs'], top_zone_1['Rho'],
                base_zone_1['Vp'], base_zone_1['Vs'], base_zone_1['Rho'],
                angles
            )
        elif avo_method == "Fatti":
            rc_1 = self.fatti_approximation(
                top_zone_1['Vp'], top_zone_1['Vs'], top_zone_1['Rho'],
                base_zone_1['Vp'], base_zone_1['Vs'], base_zone_1['Rho'],
                angles
            )
        
        # Calculate reflection coefficients for Dataset 2
        if avo_method == "Aki-Richards":
            rc_2 = self.aki_richards(
                top_zone_2['Vp'], top_zone_2['Vs'], top_zone_2['Rho'],
                base_zone_2['Vp'], base_zone_2['Vs'], base_zone_2['Rho'],
                angles
            )
        elif avo_method == "Shuey":
            rc_2 = self.shuey_approximation(
                top_zone_2['Vp'], top_zone_2['Vs'], top_zone_2['Rho'],
                base_zone_2['Vp'], base_zone_2['Vs'], base_zone_2['Rho'],
                angles
            )
        elif avo_method == "Fatti":
            rc_2 = self.fatti_approximation(
                top_zone_2['Vp'], top_zone_2['Vs'], top_zone_2['Rho'],
                base_zone_2['Vp'], base_zone_2['Vs'], base_zone_2['Rho'],
                angles
            )
        
        # Handle potential NaN values
        if np.any(np.isnan(rc_1)) or np.any(np.isnan(rc_2)):
            print("Warning: Some reflection coefficients are NaN. Check input zone data.")
            
        # --- Synthetic Seismogram - Wiggle Display (ax1 for Dataset 1) ---
        ax1.set_title(f'Synthetic 1: {top_zone_1["Zone"]}/{base_zone_1["Zone"]}', fontsize=8)
        ax1.set_xlabel('Angle (°)', fontsize=10)
        ax1.set_ylabel('Time (s)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        time = np.arange(-wavelet_duration/2, wavelet_duration/2, dt_wavelet)
        
        synthetic_traces_1 = np.zeros((len(angles), len(wavelet)))
        for i in range(len(angles)):
            synthetic_traces_1[i] = rc_1[i] * wavelet
        max_amplitude_1 = np.max(np.abs(synthetic_traces_1))
        if max_amplitude_1 == 0:
            trace_display_scale_1 = 1.0
        else:
            avg_angle_spacing = np.mean(np.diff(angles)) if len(angles) > 1 else 10
            trace_display_scale_1 = (avg_angle_spacing * 0.8) / max_amplitude_1
            
        for i, angle in enumerate(angles):
            scaled_trace_display = synthetic_traces_1[i] * trace_display_scale_1
            
            ax1.plot([angle, angle], [time.min(), time.max()], 'k:', linewidth=0.5, zorder=1)
            ax1.plot(angle + scaled_trace_display, time, 'k-', linewidth=1.5, zorder=2)
            ax1.fill_betweenx(time, angle, angle + scaled_trace_display, 
                              where=scaled_trace_display > 0, color='blue', alpha=0.5, zorder=2)
            ax1.fill_betweenx(time, angle, angle + scaled_trace_display, 
                              where=scaled_trace_display < 0, color='red', alpha=0.5, zorder=2)
        
        x_margin = (angles[1] - angles[0]) / 2 if len(angles) > 1 else 5
        ax1.set_xlim(angles.min() - x_margin, angles.max() + x_margin)
        ax1.set_ylim(time.max(), time.min())
        
        # Fine-tuning x-axis tick-labels for ax1
        if len(angles) >= 3:
            min_angle = angles[0]
            mid_angle_idx = len(angles) // 2
            mid_angle = angles[mid_angle_idx]
            max_angle = angles[-1]
            display_ticks = sorted(list(set([min_angle, mid_angle, max_angle])))
            ax1.set_xticks(display_ticks)
            ax1.set_xticklabels([f'{int(a)}°' for a in display_ticks])
        else:
            ax1.set_xticks(angles)
            ax1.set_xticklabels([f'{int(a)}°' for a in angles])
        
        ax1.tick_params(axis='x', labelsize=9)
        ax1.tick_params(axis='y', labelsize=9)
        
        # --- Synthetic Seismogram - Wiggle Display (ax2 for Dataset 2) ---
        if is_substituted:
            ax2.set_title(f'Synthetic 2: {top_zone_2["Zone"]}/{base_zone_2["Zone"]} (Fluid Substituted)', fontsize=11)
        else:
            ax2.set_title(f'Synthetic 2: {top_zone_2["Zone"]}/{base_zone_2["Zone"]}', fontsize=8)
            
        ax2.set_xlabel('Angle (°)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        synthetic_traces_2 = np.zeros((len(angles), len(wavelet)))
        for i in range(len(angles)):
            synthetic_traces_2[i] = rc_2[i] * wavelet
        max_amplitude_2 = np.max(np.abs(synthetic_traces_2))
        if max_amplitude_2 == 0:
            trace_display_scale_2 = 1.0
        else:
            avg_angle_spacing = np.mean(np.diff(angles)) if len(angles) > 1 else 10
            trace_display_scale_2 = (avg_angle_spacing * 0.8) / max_amplitude_2
            
        for i, angle in enumerate(angles):
            scaled_trace_display = synthetic_traces_2[i] * trace_display_scale_2
            
            ax2.plot([angle, angle], [time.min(), time.max()], 'k:', linewidth=0.5, zorder=1)
            ax2.plot(angle + scaled_trace_display, time, 'k-', linewidth=1.5, zorder=2)
            ax2.fill_betweenx(time, angle, angle + scaled_trace_display, 
                              where=scaled_trace_display > 0, color='blue', alpha=0.5, zorder=2)
            ax2.fill_betweenx(time, angle, angle + scaled_trace_display, 
                              where=scaled_trace_display < 0, color='red', alpha=0.5, zorder=2)
        
        ax2.set_xlim(angles.min() - x_margin, angles.max() + x_margin)
        ax2.set_ylim(time.max(), time.min())
        
        # Fine-tuning x-axis tick-labels for ax2
        if len(angles) >= 3:
            ax2.set_xticks(display_ticks)
            ax2.set_xticklabels([f'{int(a)}°' for a in display_ticks])
        else:
            ax2.set_xticks(angles)
            ax2.set_xticklabels([f'{int(a)}°' for a in angles])
        
        ax2.tick_params(axis='x', labelsize=9)
        ax2.tick_params(axis='y', labelsize=9)
        
        # --- AVO Curve Plot (ax3 for comparison) ---
        if is_substituted and original_rc_2 is not None:
            # Plot three curves: original dataset 1, original dataset 2, and substituted dataset 2
            ax3.plot(angles, rc_1, 'k-', linewidth=2, label=f'RC 1: {top_zone_1["Zone"]}/{base_zone_1["Zone"]}')
            ax3.plot(angles, original_rc_2, 'r-', linewidth=2, label=f'RC 2 (Original): {top_zone_2["Zone"]}/{base_zone_2["Zone"]}')
            ax3.plot(angles, rc_2, 'b-', linewidth=2, label=f'RC 2 (Substituted): {top_zone_2["Zone"]}/{base_zone_2["Zone"]}')
            
            if len(angles) > 0:
                # Add markers for the last angle for all curves
                ax3.plot(angles[-1], rc_1[-1], 'ko', markersize=6)
                ax3.plot(angles[-1], original_rc_2[-1], 'ro', markersize=6)
                ax3.plot(angles[-1], rc_2[-1], 'bo', markersize=6)
        else:
            # Plot two curves: original dataset 1 and dataset 2
            ax3.plot(angles, rc_1, 'k-', linewidth=2, label=f'RC 1: {top_zone_1["Zone"]}/{base_zone_1["Zone"]}')
            ax3.plot(angles, rc_2, 'r-', linewidth=2, label=f'RC 2: {top_zone_2["Zone"]}/{base_zone_2["Zone"]}')
            
            if len(angles) > 0:
                # Add markers for the last angle for both curves
                ax3.plot(angles[-1], rc_1[-1], 'ko', markersize=6)
                ax3.plot(angles[-1], rc_2[-1], 'ro', markersize=6)
        
        ax3.set_xlabel('Angle (degrees)', fontsize=10)
        ax3.set_ylabel('Reflection Coefficient', fontsize=10)
        ax3.set_title('AVO Curve Comparison', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=6)
        
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        
        # Symmetric Y-axis scaling for AVO curve (ax3)
        if is_substituted and original_rc_2 is not None:
            all_rc_values = np.concatenate((rc_1, original_rc_2, rc_2))
        else:
            all_rc_values = np.concatenate((rc_1, rc_2))
            
        max_abs_rc = np.max(np.abs(all_rc_values[~np.isnan(all_rc_values)]))
        
        y_lim_buffer = max_abs_rc * 0.1
        ax3.set_ylim(-(max_abs_rc + y_lim_buffer), (max_abs_rc + y_lim_buffer))
        ax3.tick_params(axis='x', labelsize=9)
        ax3.tick_params(axis='y', labelsize=9)
        
        # Fix y-axis scaling for synthetic seismogram panels based on AVO method
        # For Shuey, use a fixed minimum range to ensure visibility
        if avo_method == "Shuey":
            # Calculate the maximum amplitude from both synthetic traces
            all_traces = np.concatenate((synthetic_traces_1.flatten(), synthetic_traces_2.flatten()))
            max_abs_trace = np.max(np.abs(all_traces))
            
            # Set a minimum range to ensure visibility for Shuey
            min_range = 0.05  # Minimum range for y-axis
            if max_abs_trace < min_range:
                max_abs_trace = min_range
            
            # Set y-axis limits for both synthetic panels
            ax1.set_ylim(-max_abs_trace, max_abs_trace)
            ax2.set_ylim(-max_abs_trace, max_abs_trace)
        else:
            # For other methods, use the original dynamic scaling
            # This is already handled by the trace display scaling
            pass
        
        # Embed plot in tkinter
        self.synthetic_canvas = FigureCanvasTkAgg(self.synthetic_fig, master=parent_frame)
        self.synthetic_canvas.draw()
        self.synthetic_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.synthetic_canvas, parent_frame)
        toolbar.update()
    
    def generate_ricker_wavelet(self, freq: int, duration: float = 0.1, dt: float = 0.001) -> np.ndarray:
        """Generate Ricker wavelet
        
        Args:
            freq: Wavelet frequency in Hz
            duration: Wavelet duration in seconds
            dt: Time sampling in seconds
            
        Returns:
            Array with wavelet values
        """
        t = np.arange(-duration/2, duration/2, dt)
        a = (np.pi * freq * t) ** 2
        wavelet = (1 - 2*a) * np.exp(-a)
        wavelet = wavelet / np.max(np.abs(wavelet)) if np.max(np.abs(wavelet)) > 0 else wavelet
        return wavelet
        
    def aki_richards(self, vp1: float, vs1: float, rho1: float, 
                    vp2: float, vs2: float, rho2: float, 
                    theta: np.ndarray) -> np.ndarray:
        """Compute reflection coefficient using Aki-Richards approximation
        
        Args:
            vp1, vs1, rho1: Properties of upper layer
            vp2, vs2, rho2: Properties of lower layer
            theta: Array of incidence angles in degrees
            
        Returns:
            Array of reflection coefficients
        """
        if vp1 <= 0 or vs1 <= 0 or rho1 <= 0:
            return np.full_like(theta, np.nan) if isinstance(theta, np.ndarray) else np.nan
            
        theta_rad = np.radians(theta)
        k = (vs1/vp1)**2 
        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1
        
        term1 = 0.5 * (1 + np.tan(theta_rad)**2) * (dvp/vp1)
        term2 = (1 - 4*k*np.sin(theta_rad)**2) * (dvs/vs1)
        term3 = -0.5 * (4*k*np.sin(theta_rad)**2) * (drho/rho1)
        
        return term1 + term2 + term3
    
    def shuey_approximation(self, vp1: float, vs1: float, rho1: float, 
                           vp2: float, vs2: float, rho2: float, 
                           theta: np.ndarray) -> np.ndarray:
        """Compute reflection coefficient using Shuey's approximation
        
        Args:
            vp1, vs1, rho1: Properties of upper layer
            vp2, vs2, rho2: Properties of lower layer
            theta: Array of incidence angles in degrees
            
        Returns:
            Array of reflection coefficients
        """
        theta_rad = np.radians(theta)
        
        # Compute intercept and gradient
        R0 = (vp2 - vp1) / (vp2 + vp1)  # Normal incidence reflection coefficient
        
        # Compute AVO gradient (G)
        vp_avg = (vp1 + vp2) / 2
        vs_avg = (vs1 + vs2) / 2
        rho_avg = (rho1 + rho2) / 2
        
        # Shuey's three-term approximation
        term1 = R0
        term2 = R0 * (0.5 - 2 * (vs_avg/vp_avg)**2) * np.sin(theta_rad)**2
        term3 = 0.5 * (vp2 - vp1) / vp_avg * np.tan(theta_rad)**2
        
        return term1 + term2 + term3
    
    def fatti_approximation(self, vp1: float, vs1: float, rho1: float, 
                           vp2: float, vs2: float, rho2: float, 
                           theta: np.ndarray) -> np.ndarray:
        """Compute reflection coefficient using Fatti's approximation
        
        Args:
            vp1, vs1, rho1: Properties of upper layer
            vp2, vs2, rho2: Properties of lower layer
            theta: Array of incidence angles in degrees
            
        Returns:
            Array of reflection coefficients
        """
        theta_rad = np.radians(theta)
        
        # Compute impedances
        Ip1 = vp1 * rho1  # P-wave impedance
        Ip2 = vp2 * rho2
        Is1 = vs1 * rho1  # S-wave impedance
        Is2 = vs2 * rho2
        rho_avg = (rho1 + rho2) / 2
        
        # Compute reflectivities
        Rp = (Ip2 - Ip1) / (Ip2 + Ip1)
        Rs = (Is2 - Is1) / (Is2 + Is1)
        Rrho = (rho2 - rho1) / (rho2 + rho1)
        
        # Fatti's approximation
        term1 = Rp * (1 + np.tan(theta_rad)**2)
        term2 = -8 * (vs1/vp1)**2 * Rs * np.sin(theta_rad)**2
        term3 = Rrho * (0.5 * np.tan(theta_rad)**2 - 4 * (vs1/vp1)**2 * np.sin(theta_rad)**2)
        
        return term1 + term2 + term3
    
    def export_plot(self, file_path: str, plot_type: str = "log") -> Tuple[bool, str]:
        """Export current plot to file
        
        Args:
            file_path: Path to save the plot
            plot_type: Type of plot to export ("log", "synthetic", "crossplot", "3d")
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if plot_type == "log" and self.fig is not None:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                return True, f"Log plot exported to {file_path}"
            elif plot_type == "synthetic" and self.synthetic_fig is not None:
                self.synthetic_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                return True, f"Synthetic plot exported to {file_path}"
            elif plot_type == "crossplot" and self.crossplot_fig is not None:
                self.crossplot_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                return True, f"Crossplot exported to {file_path}"
            elif plot_type == "3d" and self.plot3d_fig is not None:
                self.plot3d_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                return True, f"3D plot exported to {file_path}"
            else:
                return False, "No plot available to export"
        except Exception as e:
            return False, f"Failed to export plot: {str(e)}"

# ==================== Batch Processing ====================
class BatchProcessor:
    """Batch processing capabilities for multiple wells"""
    
    def __init__(self, data_processor_class):
        self.data_processor_class = data_processor_class
        self.batch_results = {}
    
    def process_multiple_wells(self, well_files: List[str], zone_files: List[str], 
                              fluid_params: Dict[str, Dict]) -> Dict[str, Any]:
        """Process multiple wells with the same fluid substitution parameters"""
        results = {}
        
        for i, (well_file, zone_file) in enumerate(zip(well_files, zone_files)):
            well_name = os.path.splitext(os.path.basename(well_file))[0]
            print(f"Processing well {i+1}/{len(well_files)}: {well_name}")
            
            # Create a new data processor for each well
            processor = self.data_processor_class()
            
            # Load data
            success, message = processor.load_log_data(well_file)
            if not success:
                results[well_name] = {"error": message}
                continue
            
            success, message = processor.load_zone_tops(zone_file)
            if not success:
                results[well_name] = {"error": message}
                continue
            
            # Apply fluid substitutions
            fluid_results = {}
            for zone_name, fluid in fluid_params.items():
                success, message = processor.apply_fluid_substitution(
                    zone_name, 
                    fluid["type"], 
                    fluid["vp"], 
                    fluid["vs"], 
                    fluid["rho"],
                    fluid.get("method", "Gassmann")
                )
                
                if success:
                    fluid_results[zone_name] = processor.fluid_sub_zones[f"{zone_name}_{fluid['type']}"]
                else:
                    fluid_results[zone_name] = {"error": message}
            
            # Store results
            results[well_name] = {
                "blocked_data": processor.blocked_data,
                "fluid_substitutions": fluid_results,
                "elastic_moduli": processor.elastic_moduli
            }
        
        self.batch_results = results
        return results
    
    def export_batch_results(self, output_dir: str) -> bool:
        """Export batch processing results to files"""
        if not self.batch_results:
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary Excel file
        summary_path = os.path.join(output_dir, "batch_summary.xlsx")
        
        try:
            with pd.ExcelWriter(summary_path) as writer:
                # Create summary sheet
                summary_data = []
                for well_name, results in self.batch_results.items():
                    if "error" in results:
                        summary_data.append({
                            "Well": well_name,
                            "Status": "Error",
                            "Message": results["error"]
                        })
                    else:
                        summary_data.append({
                            "Well": well_name,
                            "Status": "Success",
                            "Zones Processed": len(results["fluid_substitutions"])
                        })
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
                
                # Create detailed sheets for each well
                for well_name, results in self.batch_results.items():
                    if "error" not in results:
                        # Export blocked data
                        df_blocked = pd.DataFrame(results["blocked_data"])
                        df_blocked.to_excel(writer, sheet_name=f"{well_name}_Blocked", index=False)
                        
                        # Export fluid substitution results
                        if results["fluid_substitutions"]:
                            fluid_data = []
                            for zone, data in results["fluid_substitutions"].items():
                                if "error" not in data:
                                    fluid_data.append({
                                        "Zone": zone,
                                        "Vp": data["Vp"],
                                        "Vs": data["Vs"],
                                        "Rho": data["Rho"],
                                        "FluidType": data["FluidType"],
                                        "Method": data.get("Method", "Gassmann")
                                    })
                            
                            if fluid_data:
                                pd.DataFrame(fluid_data).to_excel(writer, sheet_name=f"{well_name}_Fluid", index=False)
                        
                        # Export elastic moduli if available
                        if results["elastic_moduli"]:
                            df_moduli = pd.DataFrame(results["elastic_moduli"])
                            df_moduli.to_excel(writer, sheet_name=f"{well_name}_Moduli", index=False)
            
            return True
        
        except Exception as e:
            print(f"Error exporting batch results: {str(e)}")
            return False

# ==================== Main Application ====================
class AVOAnalyzer:
    """Main application class for AVO analysis"""
    
    def __init__(self, master):
        self.master = master
        master.title("Fluid & Lithology Analysis (©adi widyantoro 2025 - Pertamina)")
        master.geometry("1200x800")
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.viz_manager = VisualizationManager(self.data_processor, master)
        self.project_manager = ProjectManager()
        self.batch_processor = BatchProcessor(DataProcessor)
        
        # Zone selections for Synthetic Seismogram - Dataset 1
        self.selected_top_zone_1 = None
        self.selected_base_zone_1 = None
        
        # Zone selections for Synthetic Seismogram - Dataset 2
        self.selected_top_zone_2 = None
        self.selected_base_zone_2 = None
        
        # Fluid substitution state tracking
        self.is_substituted = False
        self.original_blocked_data = None
        self.original_rc_2 = None  # Store original reflection coefficients for dataset 2
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initialize plot state
        self.original_plot_state = False
               
        # Create tabs
        self.logs_tab = ttk.Frame(self.notebook)
        self.spreadsheet_tab = ttk.Frame(self.notebook)
        self.synthetic_tab = ttk.Frame(self.notebook)
        self.crossplot_tab = ttk.Frame(self.notebook)
        self.plot3d_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        self.project_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.logs_tab, text="Logs")
        self.notebook.add(self.spreadsheet_tab, text="Spreadsheet")
        self.notebook.add(self.synthetic_tab, text="Synthetic Seismogram")
        self.notebook.add(self.crossplot_tab, text="Crossplot")
        self.notebook.add(self.plot3d_tab, text="3D Visualization")
        self.notebook.add(self.batch_tab, text="Batch Processing")
        self.notebook.add(self.project_tab, text="Project Management")
        
        # Setup tabs
        self.setup_logs_tab()
        self.setup_spreadsheet_tab()
        self.setup_synthetic_tab()
        self.setup_crossplot_tab()
        self.setup_plot3d_tab()
        self.setup_batch_tab()
        self.setup_project_tab()
        
        # Setup export menu
        self.setup_export_menu()
        
        # Add status bar
        self.status_bar = ttk.Label(master, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure window resizing
        master.bind('<Configure>', self.on_resize)
    
    def setup_logs_tab(self):
        """Setup the logs tab with file input and visualization"""
        # File Input Section
        file_frame = ttk.LabelFrame(self.logs_tab, text="Well Log Data")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="Select Well Log File:").pack(side=tk.LEFT, padx=5)
        self.file_button = ttk.Button(file_frame, text="Load Logs", command=self.load_logs)
        self.file_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(file_frame, text="Select Zone Tops File:").pack(side=tk.LEFT, padx=5)
        self.zone_button = ttk.Button(file_frame, text="Load Zone Tops", command=self.load_zone_tops)
        self.zone_button.pack(side=tk.LEFT, padx=5)
        
        # Project buttons
        ttk.Button(file_frame, text="Add to Project", command=self.add_to_project).pack(side=tk.LEFT, padx=5)
        
        # Log Display Frame
        self.log_frame = ttk.Frame(self.logs_tab)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize visualization
        self.viz_manager.initialize_log_plots(self.log_frame)
    
    def setup_spreadsheet_tab(self):
        """Setup the spreadsheet tab for blocked data display"""
        # Spreadsheet Frame
        self.spreadsheet_frame = ttk.Frame(self.spreadsheet_tab)
        self.spreadsheet_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create lithology cutoff frame
        self.lithology_frame = ttk.LabelFrame(self.spreadsheet_frame, text="Lithology Cutoff Values")
        self.lithology_frame.pack(fill=tk.X, pady=5)
        
        # Vsh cutoff
        ttk.Label(self.lithology_frame, text="Vsh Cutoff:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.vsh_cutoff_var = tk.StringVar(value="0.6")
        self.vsh_cutoff_entry = ttk.Entry(self.lithology_frame, textvariable=self.vsh_cutoff_var, width=10)
        self.vsh_cutoff_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Phi cutoff
        ttk.Label(self.lithology_frame, text="Phi Cutoff:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.phi_cutoff_var = tk.StringVar(value="0.1")
        self.phi_cutoff_entry = ttk.Entry(self.lithology_frame, textvariable=self.phi_cutoff_var, width=10)
        self.phi_cutoff_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Create elastic moduli toggle
        self.moduli_frame = ttk.Frame(self.spreadsheet_frame)
        self.moduli_frame.pack(fill=tk.X, pady=5)
        
        self.show_moduli_var = tk.BooleanVar(value=False)
        ttk.Radiobutton(self.moduli_frame, text="Hide Elastic Moduli", variable=self.show_moduli_var, 
                       value=False, command=self.toggle_moduli_table).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.moduli_frame, text="Show Elastic Moduli", variable=self.show_moduli_var, 
                       value=True, command=self.toggle_moduli_table).pack(side=tk.LEFT, padx=5)
        
        # Create button frame
        self.button_frame = ttk.Frame(self.spreadsheet_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        # Add plot averages button
        self.plot_averages_button = ttk.Button(self.button_frame, text="Plot Averages on Logs", 
                                               command=self.plot_averages_on_logs, state=tk.DISABLED)
        self.plot_averages_button.pack(side=tk.LEFT, padx=5)
        
        # Add lithology button
        self.add_lithology_button = ttk.Button(self.button_frame, text="Add Lithology Column", 
                                               command=self.add_lithology_column, state=tk.DISABLED)
        self.add_lithology_button.pack(side=tk.LEFT, padx=5)
        
        # Create tables container
        self.tables_container = ttk.Frame(self.spreadsheet_frame)
        self.tables_container.pack(fill=tk.BOTH, expand=True)
        
        # Create main table frame
        self.table_frame = ttk.Frame(self.tables_container)
        self.table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create moduli table frame (initially hidden)
        self.moduli_table_frame = ttk.Frame(self.tables_container)
        # Don't pack it initially, will be shown when radio button is selected
        
        # Initial message for spreadsheet tab
        ttk.Label(self.table_frame, text="Load logs and zone tops to see blocked data here.").pack(pady=20)
    
    def setup_synthetic_tab(self):
        """Setup the synthetic seismogram tab with controls and visualization"""
        # Main Control Frame for the tab
        control_frame = ttk.LabelFrame(self.synthetic_tab, text="Synthetic Seismogram Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Use a main grid for control_frame to divide it into two columns
        control_frame.columnconfigure(0, weight=1)  # Left column
        control_frame.columnconfigure(1, weight=1)  # Right column
        
        # --- Left Panel: Dataset 1 Controls ---
        dataset1_frame = ttk.LabelFrame(control_frame, text="Dataset 1 (Original)")
        dataset1_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        dataset1_frame.columnconfigure(1, weight=1)  # Allow comboboxes to expand
        
        # Zone Selection for Dataset 1
        ttk.Label(dataset1_frame, text="Top Zone 1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.top_zone_var_1 = tk.StringVar()
        self.top_zone_combo_1 = ttk.Combobox(dataset1_frame, textvariable=self.top_zone_var_1, state="readonly")
        self.top_zone_combo_1.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.top_zone_combo_1.bind("<<ComboboxSelected>>", self.on_zone_selected)
        
        ttk.Label(dataset1_frame, text="Base Zone 1:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.base_zone_var_1 = tk.StringVar()
        self.base_zone_combo_1 = ttk.Combobox(dataset1_frame, textvariable=self.base_zone_var_1, state="readonly")
        self.base_zone_combo_1.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.base_zone_combo_1.bind("<<ComboboxSelected>>", self.on_zone_selected)
        
        # Ricker Wavelet Frequency
        ttk.Label(dataset1_frame, text="Ricker Freq (Hz):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.freq_var = tk.StringVar(value="30")
        self.freq_entry = ttk.Entry(dataset1_frame, textvariable=self.freq_var, width=10)
        self.freq_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Angle Selection
        ttk.Label(dataset1_frame, text="Angle Range (°):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        
        angle_input_frame_1 = ttk.Frame(dataset1_frame)
        angle_input_frame_1.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        angle_input_frame_1.columnconfigure(0, weight=1)
        angle_input_frame_1.columnconfigure(1, weight=1)
        
        ttk.Label(angle_input_frame_1, text="Min:").grid(row=0, column=0, sticky="w")
        self.angle_min_var = tk.StringVar(value="0")
        self.angle_min_entry = ttk.Entry(angle_input_frame_1, textvariable=self.angle_min_var, width=5)
        self.angle_min_entry.grid(row=0, column=0, padx=(30,0), pady=2, sticky="w")
        
        ttk.Label(angle_input_frame_1, text="Max:").grid(row=0, column=1, sticky="w")
        self.angle_max_var = tk.StringVar(value="30")
        self.angle_max_entry = ttk.Entry(angle_input_frame_1, textvariable=self.angle_max_var, width=5)
        self.angle_max_entry.grid(row=0, column=1, padx=(30,0), pady=2, sticky="w")
        
        # --- Right Panel: Dataset 2 Controls ---
        dataset2_frame = ttk.LabelFrame(control_frame, text="Dataset 2 (Fluid Substituted / Comparison)")
        dataset2_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        dataset2_frame.columnconfigure(1, weight=1)  # Allow comboboxes to expand
        
        # Zone Selection for Dataset 2
        ttk.Label(dataset2_frame, text="Top Zone 2:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.top_zone_var_2 = tk.StringVar()
        self.top_zone_combo_2 = ttk.Combobox(dataset2_frame, textvariable=self.top_zone_var_2, state="readonly")
        self.top_zone_combo_2.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.top_zone_combo_2.bind("<<ComboboxSelected>>", self.on_zone_selected)
        
        ttk.Label(dataset2_frame, text="Base Zone 2:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.base_zone_var_2 = tk.StringVar()
        self.base_zone_combo_2 = ttk.Combobox(dataset2_frame, textvariable=self.base_zone_var_2, state="readonly")
        self.base_zone_combo_2.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.base_zone_combo_2.bind("<<ComboboxSelected>>", self.on_zone_selected)
        
        # --- Fluid Substitution Controls ---
        fluid_frame = ttk.LabelFrame(control_frame, text="Gassmann Fluid Substitution")
        fluid_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        fluid_frame.columnconfigure(1, weight=1)
        
        ttk.Label(fluid_frame, text="Zone for Fluid Substitution:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.fluid_zone_var = tk.StringVar()
        self.fluid_zone_combo = ttk.Combobox(fluid_frame, textvariable=self.fluid_zone_var, state="readonly")
        self.fluid_zone_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.fluid_zone_combo.bind("<<ComboboxSelected>>", self.on_fluid_zone_selected)
        
        ttk.Label(fluid_frame, text="Fluid Type:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.fluid_type_var = tk.StringVar(value="Brine")
        fluid_types = ["Brine", "Oil", "Gas"]
        self.fluid_type_combo = ttk.Combobox(fluid_frame, textvariable=self.fluid_type_var, values=fluid_types, state="readonly")
        self.fluid_type_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(fluid_frame, text="Method:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.fluid_method_var = tk.StringVar(value="Gassmann")
        fluid_methods = ["Gassmann", "Xu-Payne"]
        self.fluid_method_combo = ttk.Combobox(fluid_frame, textvariable=self.fluid_method_var, values=fluid_methods, state="readonly")
        self.fluid_method_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        prop_frame = ttk.Frame(fluid_frame)
        prop_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        prop_frame.columnconfigure(1, weight=1)
        
        ttk.Label(prop_frame, text="Vp (m/s):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.vp_fluid_var = tk.StringVar(value="1500")
        self.vp_fluid_entry = ttk.Entry(prop_frame, textvariable=self.vp_fluid_var, width=10)
        self.vp_fluid_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        ttk.Label(prop_frame, text="Vs (m/s):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.vs_fluid_var = tk.StringVar(value="0")
        self.vs_fluid_entry = ttk.Entry(prop_frame, textvariable=self.vs_fluid_var, width=10)
        self.vs_fluid_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        
        ttk.Label(prop_frame, text="Rho (g/cc):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.rho_fluid_var = tk.StringVar(value="1.0")
        self.rho_fluid_entry = ttk.Entry(prop_frame, textvariable=self.rho_fluid_var, width=10)
        self.rho_fluid_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        
        # --- AVO Approximation Selection ---
        avo_frame = ttk.LabelFrame(control_frame, text="AVO Approximation")
        avo_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.avo_method_var = tk.StringVar(value="Aki-Richards")
        avo_methods = ["Aki-Richards", "Shuey", "Fatti"]
        
        for i, method in enumerate(avo_methods):
            ttk.Radiobutton(avo_frame, text=method, variable=self.avo_method_var, 
                            value=method).grid(row=0, column=i, padx=5, pady=5)
        
        # --- Generate Buttons ---
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.generate_button = ttk.Button(button_frame, text="Generate Synthetic", 
                                          command=self.generate_synthetic, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        
        self.fluid_button = ttk.Button(button_frame, text="Apply Fluid Substitution", 
                                      command=self.apply_fluid_substitution, state=tk.DISABLED)
        self.fluid_button.pack(side=tk.LEFT, padx=5)
        
        # Visualization Frame for synthetic seismogram
        self.synthetic_frame = ttk.Frame(self.synthetic_tab)
        self.synthetic_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder for synthetic plot
        ttk.Label(self.synthetic_frame, text="Select zones and generate synthetic seismogram.").pack(pady=20)
    
    def setup_crossplot_tab(self):
        """Setup the crossplot tab with controls and visualization"""
        # Control Frame
        control_frame = ttk.LabelFrame(self.crossplot_tab, text="Crossplot Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Parameter Selection
        ttk.Label(control_frame, text="X Parameter:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.x_param_var = tk.StringVar(value="Vp")
        x_params = ["Vp", "Vs", "Rho", "GR", "Phi", "Sw", "Vsh", "P-Impedance", "S-Impedance", 
                    "Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho"]
        self.x_param_combo = ttk.Combobox(control_frame, textvariable=self.x_param_var, values=x_params, state="readonly")
        self.x_param_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(control_frame, text="Y Parameter:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.y_param_var = tk.StringVar(value="Vs")
        y_params = ["Vp", "Vs", "Rho", "GR", "Phi", "Sw", "Vsh", "P-Impedance", "S-Impedance", 
                    "Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho"]
        self.y_param_combo = ttk.Combobox(control_frame, textvariable=self.y_param_var, values=y_params, state="readonly")
        self.y_param_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(control_frame, text="Color Parameter:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.color_param_var = tk.StringVar(value="Phi")
        color_params = ["None", "Vp", "Vs", "Rho", "GR", "Phi", "Sw", "Vsh", "P-Impedance", "S-Impedance", 
                    "Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho"]
        self.color_param_combo = ttk.Combobox(control_frame, textvariable=self.color_param_var, values=color_params, state="readonly")
        self.color_param_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Generate Button
        ttk.Button(control_frame, text="Generate Crossplot", 
                  command=self.generate_crossplot).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Visualization Frame
        self.crossplot_frame = ttk.Frame(self.crossplot_tab)
        self.crossplot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder for crossplot
        ttk.Label(self.crossplot_frame, text="Select parameters and generate crossplot.").pack(pady=20)
    
    def setup_plot3d_tab(self):
        """Setup the 3D visualization tab with controls and visualization"""
        # Control Frame
        control_frame = ttk.LabelFrame(self.plot3d_tab, text="3D Visualization Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Parameter Selection
        ttk.Label(control_frame, text="X Parameter:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.x3d_param_var = tk.StringVar(value="Vp")
        x3d_params = ["Vp", "Vs", "Rho", "GR", "Phi", "Sw", "Vsh", "P-Impedance", "S-Impedance", 
                      "Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho"]
        self.x3d_param_combo = ttk.Combobox(control_frame, textvariable=self.x3d_param_var, values=x3d_params, state="readonly")
        self.x3d_param_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(control_frame, text="Y Parameter:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.y3d_param_var = tk.StringVar(value="Vs")
        y3d_params = ["Vp", "Vs", "Rho", "GR", "Phi", "Sw", "Vsh", "P-Impedance", "S-Impedance", 
                      "Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho"]
        self.y3d_param_combo = ttk.Combobox(control_frame, textvariable=self.y3d_param_var, values=y3d_params, state="readonly")
        self.y3d_param_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(control_frame, text="Z Parameter:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.z3d_param_var = tk.StringVar(value="Rho")
        z3d_params = ["Vp", "Vs", "Rho", "GR", "Phi", "Sw", "Vsh", "P-Impedance", "S-Impedance", 
                      "Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho"]
        self.z3d_param_combo = ttk.Combobox(control_frame, textvariable=self.z3d_param_var, values=z3d_params, state="readonly")
        self.z3d_param_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Generate Button
        ttk.Button(control_frame, text="Generate 3D Plot", 
                  command=self.generate_3d_plot).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Visualization Frame
        self.plot3d_frame = ttk.Frame(self.plot3d_tab)
        self.plot3d_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder for 3D plot
        ttk.Label(self.plot3d_frame, text="Select parameters and generate 3D visualization.").pack(pady=20)
    
    def setup_batch_tab(self):
        """Setup the batch processing tab with controls and results"""
        # Control Frame
        control_frame = ttk.LabelFrame(self.batch_tab, text="Batch Processing Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Well Files Selection
        ttk.Label(control_frame, text="Well Log Files:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.well_files_var = tk.StringVar()
        self.well_files_entry = ttk.Entry(control_frame, textvariable=self.well_files_var, width=50)
        self.well_files_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Browse", 
                  command=self.browse_well_files).grid(row=0, column=2, padx=5, pady=5)
        
        # Zone Files Selection
        ttk.Label(control_frame, text="Zone Tops Files:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.zone_files_var = tk.StringVar()
        self.zone_files_entry = ttk.Entry(control_frame, textvariable=self.zone_files_var, width=50)
        self.zone_files_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Browse", 
                  command=self.browse_zone_files).grid(row=1, column=2, padx=5, pady=5)
        
        # Fluid Parameters
        fluid_frame = ttk.LabelFrame(control_frame, text="Fluid Substitution Parameters")
        fluid_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        ttk.Label(fluid_frame, text="Zone:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.batch_zone_var = tk.StringVar()
        self.batch_zone_entry = ttk.Entry(fluid_frame, textvariable=self.batch_zone_var, width=20)
        self.batch_zone_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(fluid_frame, text="Fluid Type:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.batch_fluid_type_var = tk.StringVar(value="Brine")
        batch_fluid_types = ["Brine", "Oil", "Gas"]
        self.batch_fluid_type_combo = ttk.Combobox(fluid_frame, textvariable=self.batch_fluid_type_var, 
                                                  values=batch_fluid_types, state="readonly")
        self.batch_fluid_type_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        ttk.Label(fluid_frame, text="Method:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.batch_fluid_method_var = tk.StringVar(value="Gassmann")
        batch_fluid_methods = ["Gassmann", "Xu-Payne"]
        self.batch_fluid_method_combo = ttk.Combobox(fluid_frame, textvariable=self.batch_fluid_method_var, 
                                                    values=batch_fluid_methods, state="readonly")
        self.batch_fluid_method_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(fluid_frame, text="Vp (m/s):").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.batch_vp_fluid_var = tk.StringVar(value="1500")
        self.batch_vp_fluid_entry = ttk.Entry(fluid_frame, textvariable=self.batch_vp_fluid_var, width=10)
        self.batch_vp_fluid_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        ttk.Label(fluid_frame, text="Vs (m/s):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.batch_vs_fluid_var = tk.StringVar(value="0")
        self.batch_vs_fluid_entry = ttk.Entry(fluid_frame, textvariable=self.batch_vs_fluid_var, width=10)
        self.batch_vs_fluid_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(fluid_frame, text="Rho (g/cc):").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.batch_rho_fluid_var = tk.StringVar(value="1.0")
        self.batch_rho_fluid_entry = ttk.Entry(fluid_frame, textvariable=self.batch_rho_fluid_var, width=10)
        self.batch_rho_fluid_entry.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        
        # Output Directory
        ttk.Label(control_frame, text="Output Directory:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.output_dir_var = tk.StringVar()
        self.output_dir_entry = ttk.Entry(control_frame, textvariable=self.output_dir_var, width=50)
        self.output_dir_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Browse", 
                  command=self.browse_output_dir).grid(row=3, column=2, padx=5, pady=5)
        
        # Process Button
        ttk.Button(control_frame, text="Process Batch", 
                  command=self.process_batch).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Results Frame
        self.batch_results_frame = ttk.Frame(self.batch_tab)
        self.batch_results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder for results
        ttk.Label(self.batch_results_frame, text="Batch processing results will appear here.").pack(pady=20)
    
    def setup_project_tab(self):
        """Setup the project management tab with controls"""
        # Control Frame
        control_frame = ttk.LabelFrame(self.project_tab, text="Project Management")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Project Creation
        ttk.Label(control_frame, text="Project Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.project_name_var = tk.StringVar()
        self.project_name_entry = ttk.Entry(control_frame, textvariable=self.project_name_var, width=30)
        self.project_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(control_frame, text="Create Project", 
                  command=self.create_project).grid(row=0, column=2, padx=5, pady=5)
        
        # Project Selection
        ttk.Label(control_frame, text="Current Project:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.current_project_var = tk.StringVar(value="None")
        self.current_project_label = ttk.Label(control_frame, textvariable=self.current_project_var)
        self.current_project_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Project List
        ttk.Label(control_frame, text="Projects:").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        
        project_list_frame = ttk.Frame(control_frame)
        project_list_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.project_listbox = tk.Listbox(project_list_frame, height=6)
        self.project_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(project_list_frame, orient=tk.VERTICAL, command=self.project_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.project_listbox.config(yscrollcommand=scrollbar.set)
        
        # Project Buttons
        project_button_frame = ttk.Frame(control_frame)
        project_button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(project_button_frame, text="Load Project", 
                  command=self.load_project).pack(side=tk.LEFT, padx=5)
        ttk.Button(project_button_frame, text="Save Project", 
                  command=self.save_project).pack(side=tk.LEFT, padx=5)
        ttk.Button(project_button_frame, text="Refresh List", 
                  command=self.refresh_project_list).pack(side=tk.LEFT, padx=5)
        
        # Project Info Frame
        self.project_info_frame = ttk.LabelFrame(self.project_tab, text="Project Information")
        self.project_info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Placeholder for project info
        ttk.Label(self.project_info_frame, text="No project loaded").pack(pady=20)
        
        # Initialize project list
        self.refresh_project_list()
    
    def setup_export_menu(self):
        """Setup export menu for plots and data"""
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Log Plot as PNG", command=lambda: self.export_plot("log", "png"))
        file_menu.add_command(label="Export Log Plot as PDF", command=lambda: self.export_plot("log", "pdf"))
        file_menu.add_command(label="Export Synthetic Plot as PNG", command=lambda: self.export_plot("synthetic", "png"))
        file_menu.add_command(label="Export Synthetic Plot as PDF", command=lambda: self.export_plot("synthetic", "pdf"))
        file_menu.add_command(label="Export Crossplot as PNG", command=lambda: self.export_plot("crossplot", "png"))
        file_menu.add_command(label="Export Crossplot as PDF", command=lambda: self.export_plot("crossplot", "pdf"))
        file_menu.add_command(label="Export 3D Plot as PNG", command=lambda: self.export_plot("3d", "png"))
        file_menu.add_command(label="Export 3D Plot as PDF", command=lambda: self.export_plot("3d", "pdf"))
        file_menu.add_separator()
        file_menu.add_command(label="Export Data as CSV", command=self.export_data_csv)
        file_menu.add_command(label="Export Data as Excel", command=self.export_data_excel)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.destroy)
    
    def load_logs(self):
        """Load well log data from file"""
        file_path = filedialog.askopenfilename(
            title="Select Well Log File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            success, message = self.data_processor.load_log_data(file_path)
            
            if success:
                # Reset plot state to original
                self.original_plot_state = False
                
                self.viz_manager.update_log_plots()
                self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"{message}\nFile: {os.path.basename(file_path)}")
                
                # Update spreadsheet to clear and disable button
                self.update_spreadsheet()
                    
            else:
                self.status_bar.config(text=f"Error: {message}")
                messagebox.showerror("Error", message)
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load logs: {str(e)}")
    
    def load_zone_tops(self):
        """Load zone tops data from file"""
        if self.data_processor.depth is None:
            messagebox.showerror("Error", "Please load well logs first.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Zone Tops File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            success, message = self.data_processor.load_zone_tops(file_path)
            
            if success:
                # Reset plot state to original
                self.original_plot_state = False
                
                # Save a copy of the original blocked data
                self.original_blocked_data = copy.deepcopy(self.data_processor.blocked_data)
                
                self.viz_manager.update_log_plots()
                self.update_spreadsheet()
                self.update_zone_comboboxes()
                self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"{message}\nFile: {os.path.basename(file_path)}")
                  
            else:
                self.status_bar.config(text=f"Error: {message}")
                messagebox.showerror("Error", message)
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load zone tops: {str(e)}")
    
    def add_to_project(self):
        """Add current files to project"""
        if not self.project_manager.current_project:
            messagebox.showerror("Error", "No project loaded. Please create or load a project first.")
            return
        
        # Add log file if loaded
        if hasattr(self, 'last_log_file') and self.last_log_file:
            self.project_manager.add_file_to_project(self.last_log_file, 'logs')
        
        # Add zone file if loaded
        if hasattr(self, 'last_zone_file') and self.last_zone_file:
            self.project_manager.add_file_to_project(self.last_zone_file, 'zones')
        
        messagebox.showinfo("Success", "Files added to project successfully.")
    
    def update_zone_comboboxes(self):
        """Update zone selection comboboxes with available zones"""
        if self.data_processor.blocked_data is None or not self.data_processor.blocked_data['Zone']:
            zone_names = []
        else:
            zone_names = self.data_processor.blocked_data['Zone']
        
        # Update Dataset 1 Comboboxes
        self.top_zone_combo_1['values'] = zone_names
        self.base_zone_combo_1['values'] = zone_names
        
        # Update Dataset 2 Comboboxes
        self.top_zone_combo_2['values'] = zone_names
        self.base_zone_combo_2['values'] = zone_names
        
        # Update Fluid Zone Combobox
        self.fluid_zone_combo['values'] = zone_names
        
        # Set default selections for Dataset 1
        if len(zone_names) > 0:
            self.top_zone_var_1.set(zone_names[0])
            if len(zone_names) > 1:
                self.base_zone_var_1.set(zone_names[min(1, len(zone_names) -1)])
            else:
                self.base_zone_var_1.set(zone_names[0])
            
            # Set default selections for Dataset 2 (initially same as Dataset 1)
            self.top_zone_var_2.set(self.top_zone_var_1.get())
            self.base_zone_var_2.set(self.base_zone_var_1.get())
            
            # Set default for Fluid Zone
            self.fluid_zone_var.set(zone_names[0])
            self.on_zone_selected(None)  # Manually trigger to enable/disable button
        else:
            self.top_zone_var_1.set("")
            self.base_zone_var_1.set("")
            self.top_zone_var_2.set("")
            self.base_zone_var_2.set("")
            self.fluid_zone_var.set("")
            self.generate_button.config(state=tk.DISABLED)
            self.fluid_button.config(state=tk.DISABLED)
    
    def update_spreadsheet(self):
        """Update spreadsheet with blocked data"""
        # Clear the table frame content but keep the structure
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Display the data
        if self.data_processor.blocked_data is None or not self.data_processor.blocked_data['Zone']:
            ttk.Label(self.table_frame, text="No blocked data available. Load logs and zone tops to see data here.").pack(pady=20)
            
            # Enable/disable buttons based on data availability
            self.plot_averages_button.config(state=tk.DISABLED)
            self.add_lithology_button.config(state=tk.DISABLED)
            return
            
        df = pd.DataFrame(self.data_processor.blocked_data)
        
        try:
            self.table = Table(self.table_frame, dataframe=df, showtoolbar=False, showstatusbar=False)
            self.table.show()
        except Exception as e:
            messagebox.showwarning("PandasTable Warning", 
                                   f"Failed to load full interactive table: {str(e)}\n"
                                   "Displaying data as plain text. Please ensure 'pandastable' is installed correctly (`pip install pandastable`).")
            text_frame = ttk.Frame(self.table_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            text = tk.Text(text_frame, wrap=tk.NONE)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text.insert(tk.END, df.to_string(index=False))
            
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.config(yscrollcommand=scrollbar.set)
        
        # Enable/disable buttons based on data availability
        self.plot_averages_button.config(state=tk.NORMAL)
        self.add_lithology_button.config(state=tk.NORMAL)

    def plot_averages_on_logs(self):
        """Plot the average values from blocked data as constant lines on the log plots"""
        if self.data_processor.blocked_data is None or not self.data_processor.blocked_data['Zone']:
            messagebox.showwarning("No Data", "No blocked data available to plot")
            return
        
        if self.data_processor.zone_tops is None:
            messagebox.showwarning("No Data", "No zone tops data available to plot")
            return
        
        # Switch to Logs tab
        self.notebook.select(0)  # Logs tab is the first tab
        
        # Store the current state to allow resetting
        self.original_plot_state = True
        
        # Plot the averages
        self.viz_manager.plot_blocked_averages()
        
        self.status_bar.config(text="Average values plotted as constant lines on logs")
        messagebox.showinfo("Success", "Average values plotted as constant lines on log plots")
   
    def add_lithology_column(self):
        """Add lithology column to the spreadsheet based on cutoff values"""
        if self.data_processor.blocked_data is None or not self.data_processor.blocked_data['Zone']:
            messagebox.showwarning("No Data", "No blocked data available")
            return
        
        try:
            # Get cutoff values from user input
            vsh_cutoff = float(self.vsh_cutoff_var.get())
            phi_cutoff = float(self.phi_cutoff_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric cutoff values")
            return
        
        # Calculate lithology for each zone
        lithology_values = []
        for i in range(len(self.data_processor.blocked_data['Zone'])):
            vsh = self.data_processor.blocked_data['Vsh'][i]
            phi = self.data_processor.blocked_data['Phi'][i]
            
            if vsh <= vsh_cutoff and phi >= phi_cutoff:
                lithology_values.append("Sd")  # Sand
            else:
                lithology_values.append("Sh")  # Shale
        
        # Add lithology values to blocked data
        self.data_processor.blocked_data['Lithology'] = lithology_values
        
        # Update the spreadsheet display
        self.update_spreadsheet()
        
        # If moduli table is visible, update it too
        if self.show_moduli_var.get():
            self.update_moduli_table()
        
        self.status_bar.config(text="Lithology column added to spreadsheet")
        messagebox.showinfo("Success", "Lithology column added to spreadsheet")

    def toggle_moduli_table(self):
        """Show or hide the elastic moduli table based on radio button selection"""
        if self.show_moduli_var.get():
            # Show the moduli table
            self.moduli_table_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            self.update_moduli_table()
        else:
            # Hide the moduli table
            self.moduli_table_frame.pack_forget()

            
    def update_moduli_table(self):
        """Update the elastic moduli table with current data"""
        # Clear previous table
        for widget in self.moduli_table_frame.winfo_children():
            widget.destroy()
        
        if self.data_processor.blocked_data is None or not self.data_processor.blocked_data['Zone']:
            ttk.Label(self.moduli_table_frame, text="No data available for elastic moduli calculation.").pack(pady=20)
            return
        
        # Create elastic moduli data
        moduli_data = {
            'Zone': self.data_processor.blocked_data['Zone']
        }
        
        # Calculate elastic moduli for each zone
        for i, zone in enumerate(self.data_processor.blocked_data['Zone']):
            # Get zone data
            zone_data = self.data_processor.get_zone_data(zone)
            
            if zone_data is None:
                continue
            
            # Calculate elastic moduli
            vp = zone_data['Vp']
            vs = zone_data['Vs']
            rho = zone_data['Rho'] * 1000  # Convert to kg/m3
            
            # Bulk modulus
            k = rho * (vp**2 - 4/3 * vs**2)
            
            # Shear modulus
            mu = rho * vs**2
            
            # Young's modulus
            with np.errstate(divide='ignore', invalid='ignore'):
                E = np.where((3 * k + mu) != 0, 9 * k * mu / (3 * k + mu), np.nan)
            
            # Poisson's ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                pr = np.where((3 * k + mu) != 0, (3 * k - 2 * mu) / (2 * (3 * k + mu)), np.nan)
            
            # Lambda parameter
            lam = k - 2/3 * mu
            
            # P-wave and S-wave impedances
            ip = vp * rho
            is_ = vs * rho
            
            # Add to moduli data if not already present
            if 'Bulk Modulus' not in moduli_data:
                moduli_data['Bulk Modulus'] = []
                moduli_data['Shear Modulus'] = []
                moduli_data['Youngs Modulus'] = []
                moduli_data['Poissons Ratio'] = []
                moduli_data['Lambda'] = []
                moduli_data['P-Impedance'] = []
                moduli_data['S-Impedance'] = []
            
            # Append values
            moduli_data['Bulk Modulus'].append(k)
            moduli_data['Shear Modulus'].append(mu)
            moduli_data['Youngs Modulus'].append(E)
            moduli_data['Poissons Ratio'].append(pr)
            moduli_data['Lambda'].append(lam)
            moduli_data['P-Impedance'].append(ip)
            moduli_data['S-Impedance'].append(is_)
        
        # Create DataFrame and display table
        df_moduli = pd.DataFrame(moduli_data)
        
        try:
            moduli_table = Table(self.moduli_table_frame, dataframe=df_moduli, showtoolbar=False, showstatusbar=False)
            moduli_table.show()
        except Exception as e:
            messagebox.showwarning("PandasTable Warning", 
                                   f"Failed to load full interactive moduli table: {str(e)}\n"
                                   "Displaying data as plain text. Please ensure 'pandastable' is installed correctly (`pip install pandastable`).")
            text_frame = ttk.Frame(self.moduli_table_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            text = tk.Text(text_frame, wrap=tk.NONE)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text.insert(tk.END, df_moduli.to_string(index=False))
            
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.config(yscrollcommand=scrollbar.set)
        
        # Store moduli data for use in other tabs
        self.elastic_moduli_data = moduli_data
        
        # Also update the data_processor's elastic_moduli for compatibility
        self.data_processor.elastic_moduli = {
            'Bulk Modulus': np.array(moduli_data['Bulk Modulus']),
            'Shear Modulus': np.array(moduli_data['Shear Modulus']),
            'Youngs Modulus': np.array(moduli_data['Youngs Modulus']),
            'Poissons Ratio': np.array(moduli_data['Poissons Ratio']),
            'Lambda': np.array(moduli_data['Lambda']),
            'P-Impedance': np.array(moduli_data['P-Impedance']),
            'S-Impedance': np.array(moduli_data['S-Impedance'])
        }

    def on_zone_selected(self, event):
        """Handle zone selection changes for both datasets"""
        self.selected_top_zone_1 = self.top_zone_var_1.get()
        self.selected_base_zone_1 = self.base_zone_var_1.get()
        self.selected_top_zone_2 = self.top_zone_var_2.get()
        self.selected_base_zone_2 = self.base_zone_var_2.get()
        
        can_generate_1 = False
        can_generate_2 = False
        
        if self.data_processor.blocked_data and self.data_processor.blocked_data['Zone']:
            zone_names = self.data_processor.blocked_data['Zone']
            
            # Validate Dataset 1 zones
            if self.selected_top_zone_1 and self.selected_base_zone_1:
                try:
                    top_idx_1 = zone_names.index(self.selected_top_zone_1)
                    base_idx_1 = zone_names.index(self.selected_base_zone_1)
                    if top_idx_1 < base_idx_1:
                        can_generate_1 = True
                except ValueError:
                    pass  # Zones not found, keep can_generate_1 as False
            
            # Validate Dataset 2 zones
            if self.selected_top_zone_2 and self.selected_base_zone_2:
                try:
                    top_idx_2 = zone_names.index(self.selected_top_zone_2)
                    base_idx_2 = zone_names.index(self.selected_base_zone_2)
                    if top_idx_2 < base_idx_2:
                        can_generate_2 = True
                except ValueError:
                    pass  # Zones not found, keep can_generate_2 as False
        
        # Enable main generate button only if BOTH datasets have valid selections
        if can_generate_1 and can_generate_2:
            self.generate_button.config(state=tk.NORMAL)
            self.fluid_button.config(state=tk.NORMAL)
        else:
            self.generate_button.config(state=tk.DISABLED)
            self.fluid_button.config(state=tk.DISABLED)
        
        # Provide specific feedback if one pair is invalid
        if (self.selected_top_zone_1 and self.selected_base_zone_1 and not can_generate_1):
             messagebox.showwarning("Selection Error", "Dataset 1: 'Top Zone' must be stratigraphically above 'Base Zone'.")
        if (self.selected_top_zone_2 and self.selected_base_zone_2 and not can_generate_2):
             messagebox.showwarning("Selection Error", "Dataset 2: 'Top Zone' must be stratigraphically above 'Base Zone'.")
    
    def on_fluid_zone_selected(self, event):
        """Handle fluid zone selection changes"""
        # This method can be expanded later if you want to update fluid properties
        # based on the selected zone's initial properties.
        pass
    
    def validate_inputs(self):
        """Validate user inputs before generating synthetic seismogram"""
        try:
            freq = int(self.freq_var.get())
            if not (10 <= freq <= 80):
                messagebox.showerror("Error", "Ricker Wavelet Frequency must be an integer between 10 and 80 Hz.")
                return None
            
            min_angle = int(self.angle_min_var.get())
            max_angle = int(self.angle_max_var.get())
            
            if not (0 <= min_angle <= 60) or not (0 <= max_angle <= 60):
                messagebox.showerror("Error", "Angle range values must be integers between 0 and 60 degrees.")
                return None
            
            if min_angle >= max_angle:
                messagebox.showerror("Error", "Minimum angle must be less than maximum angle.")
                return None
            
            angles = np.arange(min_angle, max_angle + 1)
            
            if len(angles) > 30:
                angles = np.linspace(min_angle, max_angle, 30).astype(int)
            
            return freq, angles
            
        except ValueError:
            messagebox.showerror("Error", "All inputs (Frequency, Min Angle, Max Angle) must be valid integers.")
            return None
    
    def generate_synthetic(self):
        """Generate synthetic seismogram for selected zones"""
        # If we're in a substituted state, revert to original data
        if self.is_substituted:
            self.data_processor.blocked_data = copy.deepcopy(self.original_blocked_data)
            self.is_substituted = False
            self.original_rc_2 = None
            
            # Update displays to show original data
            self.viz_manager.update_log_plots()
            self.update_spreadsheet()
        
        if self.data_processor.blocked_data is None or not self.data_processor.blocked_data['Zone']:
            messagebox.showerror("Error", "No valid blocked data available. Please load logs and zone tops first.")
            return
        
        # Validate inputs (frequency and angles)
        validated_inputs = self.validate_inputs()
        if validated_inputs is None:
            return
        
        freq, angles = validated_inputs
        
        # Get selected zone data for Dataset 1
        top_zone_data_1 = self.data_processor.get_zone_data(self.selected_top_zone_1)
        base_zone_data_1 = self.data_processor.get_zone_data(self.selected_base_zone_1)
        
        if top_zone_data_1 is None or base_zone_data_1 is None:
            messagebox.showerror("Data Error", "Dataset 1: Selected zones not found in blocked data. Try reloading data.")
            return
        
        # Check for NaN values in selected zone data for Dataset 1 (only numeric fields)
        numeric_keys = ['Vp', 'Vs', 'Rho', 'Depth', 'GR', 'Phi', 'Sw', 'Vsh']
        if any(np.isnan(top_zone_data_1[key]) for key in numeric_keys) or \
           any(np.isnan(base_zone_data_1[key]) for key in numeric_keys):
            messagebox.showerror("Data Error", "Dataset 1: Selected zones contain missing (NaN) values for required numeric properties. Cannot generate synthetic seismogram.")
            return
        
        # Get selected zone data for Dataset 2
        top_zone_data_2 = self.data_processor.get_zone_data(self.selected_top_zone_2)
        base_zone_data_2 = self.data_processor.get_zone_data(self.selected_base_zone_2)
        
        if top_zone_data_2 is None or base_zone_data_2 is None:
            messagebox.showerror("Data Error", "Dataset 2: Selected zones not found in blocked data. Try reloading data.")
            return
        
        # Check for NaN values in selected zone data for Dataset 2 (only numeric fields)
        if any(np.isnan(top_zone_data_2[key]) for key in numeric_keys) or \
           any(np.isnan(base_zone_data_2[key]) for key in numeric_keys):
            messagebox.showerror("Data Error", "Dataset 2: Selected zones contain missing (NaN) values for required numeric properties. Cannot generate synthetic seismogram.")
            return
        
        # Get AVO method
        avo_method = self.avo_method_var.get()
        
        # Generate synthetic seismogram
        self.viz_manager.create_synthetic_seismogram(
            self.synthetic_frame, top_zone_data_1, base_zone_data_1, 
            top_zone_data_2, base_zone_data_2, freq, angles, avo_method,
            self.is_substituted, self.original_rc_2
        )
        
        self.status_bar.config(text="Synthetic seismogram generated successfully")
        messagebox.showinfo("Success", "Synthetic seismograms generated successfully.")
    
    def apply_fluid_substitution(self):
        """Applies Gassmann fluid substitution to the selected zone"""
        if self.data_processor.blocked_data is None or not self.data_processor.blocked_data['Zone']:
            messagebox.showerror("Error", "No blocked data available. Please load logs and zone tops first.")
            return
            
        # Get selected zone
        selected_zone = self.fluid_zone_var.get()
        if not selected_zone:
            messagebox.showerror("Error", "Please select a zone for fluid substitution.")
            return
            
        # Get fluid properties
        try:
            vp_fluid = float(self.vp_fluid_var.get())
            vs_fluid = float(self.vs_fluid_var.get())
            rho_fluid = float(self.rho_fluid_var.get())
            fluid_type = self.fluid_type_var.get()
            fluid_method = self.fluid_method_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid fluid properties. Please enter numeric values.")
            return
            
        # If we're not in a substituted state, save the current reflection coefficients for dataset 2
        if not self.is_substituted:
            # Calculate and save the original reflection coefficients for dataset 2
            freq, angles = self.validate_inputs()
            if freq is None:
                return
                
            # Get the current zone data for dataset 2
            top_zone_data_2 = self.data_processor.get_zone_data(self.selected_top_zone_2)
            base_zone_data_2 = self.data_processor.get_zone_data(self.selected_base_zone_2)
            
            if top_zone_data_2 is None or base_zone_data_2 is None:
                messagebox.showerror("Data Error", "Dataset 2: Selected zones not found in blocked data. Try reloading data.")
                return
                
            # Calculate the original reflection coefficients
            avo_method = self.avo_method_var.get()
            if avo_method == "Aki-Richards":
                self.original_rc_2 = self.viz_manager.aki_richards(
                    top_zone_data_2['Vp'], top_zone_data_2['Vs'], top_zone_data_2['Rho'],
                    base_zone_data_2['Vp'], base_zone_data_2['Vs'], base_zone_data_2['Rho'],
                    angles
                )
            elif avo_method == "Shuey":
                self.original_rc_2 = self.viz_manager.shuey_approximation(
                    top_zone_data_2['Vp'], top_zone_data_2['Vs'], top_zone_data_2['Rho'],
                    base_zone_data_2['Vp'], base_zone_data_2['Vs'], base_zone_data_2['Rho'],
                    angles
                )
            elif avo_method == "Fatti":
                self.original_rc_2 = self.viz_manager.fatti_approximation(
                    top_zone_data_2['Vp'], top_zone_data_2['Vs'], top_zone_data_2['Rho'],
                    base_zone_data_2['Vp'], base_zone_data_2['Vs'], base_zone_data_2['Rho'],
                    angles
                )
        
        # Apply fluid substitution
        success, message = self.data_processor.apply_fluid_substitution(
            selected_zone, fluid_type, vp_fluid, vs_fluid, rho_fluid, fluid_method
        )
        
        if success:
            # Set the substitution flag
            self.is_substituted = True
            
            # Remove lithology column if it exists (since data has changed)
            if self.data_processor.blocked_data and 'Lithology' in self.data_processor.blocked_data:
                self.data_processor.blocked_data.pop('Lithology')
                self.update_spreadsheet()
            
            # Clear elastic moduli data if it exists (since data has changed)
            if hasattr(self, 'elastic_moduli_data'):
                self.elastic_moduli_data = None
                if self.show_moduli_var.get():
                    self.update_moduli_table()
            
            # Get the updated zone data for dataset 2 (after substitution)
            top_zone_data_2 = self.data_processor.get_zone_data(self.selected_top_zone_2)
            base_zone_data_2 = self.data_processor.get_zone_data(self.selected_base_zone_2)
            
            # Get current parameters for synthetic generation
            freq, angles = self.validate_inputs()
            avo_method = self.avo_method_var.get()
            
            # Generate synthetic seismogram with the substituted data
            self.viz_manager.create_synthetic_seismogram(
                self.synthetic_frame, 
                self.data_processor.get_zone_data(self.selected_top_zone_1), 
                self.data_processor.get_zone_data(self.selected_base_zone_1),
                top_zone_data_2, base_zone_data_2, freq, angles, avo_method,
                self.is_substituted, self.original_rc_2
            )
            
            self.status_bar.config(text=f"Fluid substitution applied to {selected_zone}")
            messagebox.showinfo("Success", message)
        else:
            self.status_bar.config(text=f"Error: {message}")
            messagebox.showerror("Error", message)
    
    def generate_crossplot(self):
        """Generate crossplot visualization"""
        x_param = self.x_param_var.get()
        y_param = self.y_param_var.get()
        color_param = self.color_param_var.get()
        
        if color_param == "None":
            color_param = None
        
        # Check if we need to use elastic moduli data
        if x_param in ["Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho", "P-Impedance", "S-Impedance"] or \
           y_param in ["Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho", "P-Impedance", "S-Impedance"]:
            if not hasattr(self, 'elastic_moduli_data') or not self.elastic_moduli_data:
                messagebox.showwarning("No Data", "Elastic moduli data not available. Please show and update the elastic moduli table first.")
                return
        
        self.viz_manager.create_crossplot(self.crossplot_frame, x_param, y_param, color_param)
        self.status_bar.config(text=f"Crossplot generated: {x_param} vs {y_param}")
    
    def generate_3d_plot(self):
        """Generate 3D visualization"""
        x_param = self.x3d_param_var.get()
        y_param = self.y3d_param_var.get()
        z_param = self.z3d_param_var.get()
        
        # Check if we need to use elastic moduli data
        if x_param in ["Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho", "P-Impedance", "S-Impedance"] or \
           y_param in ["Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho", "P-Impedance", "S-Impedance"] or \
           z_param in ["Bulk Modulus", "Shear Modulus", "Youngs Modulus", "Poissons Ratio", "Lambda", "MuRho", "P-Impedance", "S-Impedance"]:
            if not hasattr(self, 'elastic_moduli_data') or not self.elastic_moduli_data:
                messagebox.showwarning("No Data", "Elastic moduli data not available. Please show and update the elastic moduli table first.")
                return
        
        self.viz_manager.create_3d_visualization(self.plot3d_frame, x_param, y_param, z_param)
        self.status_bar.config(text=f"3D plot generated: {x_param}, {y_param}, {z_param}")
    
    def browse_well_files(self):
        """Browse for well log files"""
        file_paths = filedialog.askopenfilenames(
            title="Select Well Log Files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_paths:
            self.well_files_var.set(";".join(file_paths))
    
    def browse_zone_files(self):
        """Browse for zone tops files"""
        file_paths = filedialog.askopenfilenames(
            title="Select Zone Tops Files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_paths:
            self.zone_files_var.set(";".join(file_paths))
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        
        if directory:
            self.output_dir_var.set(directory)
    
    def process_batch(self):
        """Process batch of wells"""
        # Get input parameters
        well_files = self.well_files_var.get().split(";")
        zone_files = self.zone_files_var.get().split(";")
        output_dir = self.output_dir_var.get()
        
        if not well_files or not zone_files or not output_dir:
            messagebox.showerror("Error", "Please select well files, zone files, and output directory.")
            return
        
        if len(well_files) != len(zone_files):
            messagebox.showerror("Error", "Number of well files must match number of zone files.")
            return
        
        # Get fluid parameters
        try:
            zone = self.batch_zone_var.get()
            fluid_type = self.batch_fluid_type_var.get()
            fluid_method = self.batch_fluid_method_var.get()
            vp_fluid = float(self.batch_vp_fluid_var.get())
            vs_fluid = float(self.batch_vs_fluid_var.get())
            rho_fluid = float(self.batch_rho_fluid_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid fluid parameters. Please check your input.")
            return
        
        # Create fluid parameters dictionary
        fluid_params = {
            zone: {
                "type": fluid_type,
                "method": fluid_method,
                "vp": vp_fluid,
                "vs": vs_fluid,
                "rho": rho_fluid
            }
        }
        
        # Process batch
        try:
            results = self.batch_processor.process_multiple_wells(well_files, zone_files, fluid_params)
            
            # Export results
            success = self.batch_processor.export_batch_results(output_dir)
            
            if success:
                # Display results in the UI
                self.display_batch_results(results)
                self.status_bar.config(text="Batch processing completed successfully")
                messagebox.showinfo("Success", "Batch processing completed successfully.")
            else:
                self.status_bar.config(text="Error: Failed to export batch results")
                messagebox.showerror("Error", "Failed to export batch results.")
        
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")
    
    def display_batch_results(self, results):
        """Display batch processing results in the UI"""
        # Clear previous results
        for widget in self.batch_results_frame.winfo_children():
            widget.destroy()
        
        # Create results text widget
        results_text = tk.Text(self.batch_results_frame, wrap=tk.WORD)
        results_text.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.batch_results_frame, orient=tk.VERTICAL, command=results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.config(yscrollcommand=scrollbar.set)
        
        # Display results
        for well_name, result in results.items():
            if "error" in result:
                results_text.insert(tk.END, f"Well: {well_name}\n")
                results_text.insert(tk.END, f"Status: Error\n")
                results_text.insert(tk.END, f"Message: {result['error']}\n\n")
            else:
                results_text.insert(tk.END, f"Well: {well_name}\n")
                results_text.insert(tk.END, f"Status: Success\n")
                results_text.insert(tk.END, f"Zones Processed: {len(result['fluid_substitutions'])}\n\n")
                
                # Display fluid substitution results
                for zone, data in result['fluid_substitutions'].items():
                    if "error" not in data:
                        results_text.insert(tk.END, f"  {zone}: {data['FluidType']} ({data.get('Method', 'Gassmann')})\n")
                        results_text.insert(tk.END, f"    Vp: {data['Vp']:.2f} m/s\n")
                        results_text.insert(tk.END, f"    Vs: {data['Vs']:.2f} m/s\n")
                        results_text.insert(tk.END, f"    Rho: {data['Rho']:.2f} g/cc\n\n")
        
        results_text.config(state=tk.DISABLED)
    
    def create_project(self):
        """Create a new project"""
        project_name = self.project_name_var.get().strip()
        
        if not project_name:
            messagebox.showerror("Error", "Please enter a project name.")
            return
        
        success = self.project_manager.create_project(project_name)
        
        if success:
            self.current_project_var.set(project_name)
            self.refresh_project_list()
            self.update_project_info()
            self.status_bar.config(text=f"Project created: {project_name}")
            messagebox.showinfo("Success", f"Project '{project_name}' created successfully.")
        else:
            self.status_bar.config(text="Error: Failed to create project")
            messagebox.showerror("Error", "Failed to create project. Project name may already exist.")
    
    def load_project(self):
        """Load an existing project"""
        selection = self.project_listbox.curselection()
        
        if not selection:
            messagebox.showerror("Error", "Please select a project to load.")
            return
        
        project_name = self.project_listbox.get(selection[0])
        
        success = self.project_manager.load_project(project_name)
        
        if success:
            self.current_project_var.set(project_name)
            self.update_project_info()
            self.status_bar.config(text=f"Project loaded: {project_name}")
            messagebox.showinfo("Success", f"Project '{project_name}' loaded successfully.")
        else:
            self.status_bar.config(text="Error: Failed to load project")
            messagebox.showerror("Error", "Failed to load project.")
    
    def save_project(self):
        """Save the current project"""
        if not self.project_manager.current_project:
            messagebox.showerror("Error", "No project loaded.")
            return
        
        success = self.project_manager.save_project()
        
        if success:
            self.status_bar.config(text=f"Project saved: {self.project_manager.current_project}")
            messagebox.showinfo("Success", "Project saved successfully.")
        else:
            self.status_bar.config(text="Error: Failed to save project")
            messagebox.showerror("Error", "Failed to save project.")
    
    def refresh_project_list(self):
        """Refresh the project list"""
        # Clear current list
        self.project_listbox.delete(0, tk.END)
        
        # Get list of project directories
        try:
            project_dirs = [d for d in os.listdir(self.project_manager.project_dir) 
                           if os.path.isdir(os.path.join(self.project_manager.project_dir, d))]
            
            # Add projects to listbox
            for project in project_dirs:
                self.project_listbox.insert(tk.END, project)
                
                # Update projects dictionary
                if project not in self.project_manager.projects:
                    self.project_manager.projects[project] = {
                        'path': os.path.join(self.project_manager.project_dir, project),
                        'log_files': [],
                        'zone_files': [],
                        'settings': {}
                    }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh project list: {str(e)}")
    
    def update_project_info(self):
        """Update project information display"""
        # Clear previous info
        for widget in self.project_info_frame.winfo_children():
            widget.destroy()
        
        if not self.project_manager.current_project:
            ttk.Label(self.project_info_frame, text="No project loaded").pack(pady=20)
            return
        
        project = self.project_manager.projects[self.project_manager.current_project]
        
        # Display project information
        info_frame = ttk.Frame(self.project_info_frame)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(info_frame, text=f"Project: {self.project_manager.current_project}", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        
        ttk.Label(info_frame, text="Path:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(info_frame, text=project['path']).grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(info_frame, text="Created:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(info_frame, text=project.get('created_at', 'Unknown')).grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(info_frame, text="Log Files:").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(info_frame, text="\n".join(project['log_files'])).grid(row=3, column=1, sticky="w", pady=2)
        
        ttk.Label(info_frame, text="Zone Files:").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(info_frame, text="\n".join(project['zone_files'])).grid(row=4, column=1, sticky="w", pady=2)
    
    def export_plot(self, plot_type, format):
        """Export plot to file"""
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, "text")
        
        # Check if the current tab matches the plot type
        tab_to_plot_type = {
            "Logs": "log",
            "Synthetic Seismogram": "synthetic",
            "Crossplot": "crossplot",
            "3D Visualization": "3d"
        }
        
        if tab_text in tab_to_plot_type and tab_to_plot_type[tab_text] != plot_type:
            messagebox.showinfo("Export Error", 
                              f"Please switch to the {tab_text} tab to export {plot_type} plots.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title=f"Export {plot_type.title()} Plot as {format.upper()}",
            defaultextension=f".{format}",
            filetypes=[(f"{format.upper()} files", f"*.{format}"), ("All files", "*.*")]
        )
        
        if file_path:
            success, message = self.viz_manager.export_plot(file_path, plot_type)
            if success:
                self.status_bar.config(text=f"Exported: {os.path.basename(file_path)}")
                messagebox.showinfo("Export Successful", message)
            else:
                self.status_bar.config(text=f"Export error: {message}")
                messagebox.showerror("Export Error", message)
    
    def export_data_csv(self):
        """Export blocked data as CSV"""
        file_path = filedialog.asksaveasfilename(
            title="Export Data as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            success, message = self.data_processor.export_blocked_data(file_path)
            if success:
                self.status_bar.config(text=f"Exported: {os.path.basename(file_path)}")
                messagebox.showinfo("Export Successful", message)
            else:
                self.status_bar.config(text=f"Export error: {message}")
                messagebox.showerror("Export Error", message)
    
    def export_data_excel(self):
        """Export blocked data as Excel"""
        file_path = filedialog.asksaveasfilename(
            title="Export Data as Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            success, message = self.data_processor.export_to_excel(file_path)
            if success:
                self.status_bar.config(text=f"Exported: {os.path.basename(file_path)}")
                messagebox.showinfo("Export Successful", message)
            else:
                self.status_bar.config(text=f"Export error: {message}")
                messagebox.showerror("Export Error", message)
    
    def on_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.master:
            current_tab_text = self.notebook.tab(self.notebook.select(), "text")
            
            if current_tab_text == "Logs" and hasattr(self.viz_manager, 'canvas') and self.viz_manager.canvas is not None:
                self.viz_manager.canvas.draw_idle()
            elif current_tab_text == "Synthetic Seismogram" and hasattr(self.viz_manager, 'synthetic_canvas') and self.viz_manager.synthetic_canvas is not None:
                self.viz_manager.synthetic_canvas.draw_idle()
            elif current_tab_text == "Crossplot" and hasattr(self.viz_manager, 'crossplot_canvas') and self.viz_manager.crossplot_canvas is not None:
                self.viz_manager.crossplot_canvas.draw_idle()
            elif current_tab_text == "3D Visualization" and hasattr(self.viz_manager, 'plot3d_canvas') and self.viz_manager.plot3d_canvas is not None:
                self.viz_manager.plot3d_canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = AVOAnalyzer(root)
    root.mainloop()