import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum, auto

class ConfigErrorType(Enum):
    """Types of configuration errors"""
    FILE_NOT_FOUND = auto()
    INVALID_FORMAT = auto()
    MISSING_SECTION = auto()
    INVALID_VALUE = auto()
    VERSION_MISMATCH = auto()

@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: Dict[ConfigErrorType, List[str]]
    warnings: List[str]

class ConfigLoader:
    """
    Professional-Grade Configuration Loader with:
    - Multi-format support (YAML/JSON/ENV)
    - Schema validation
    - Environment-aware loading
    - Version control
    - Comprehensive error handling
    - Default value management
    - Hot-reloading capability
    """
    
    DEFAULT_CONFIG_PATHS = [
        "config/parameters.yaml",
        "config/parameters.json",
        "config/.env",
        "/etc/trading_system/config.yaml"
    ]
    
    CONFIG_SCHEMA = {
        'required_sections': ['strategy', 'execution', 'monitoring'],
        'version': {'min': 1.0, 'max': 2.0},
        'value_constraints': {
            'execution.max_retries': {'min': 1, 'max': 10},
            'strategy.risk.max_position_pct': {'min': 0.01, 'max': 0.5}
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = None
        self._last_modified = 0
        
    def load(self, config_path: Optional[str] = None, 
            env: str = "production", 
            validate: bool = True) -> Dict[str, Any]:
        """
        Load configuration with advanced features
        Args:
            config_path: Explicit path to config file. If None, searches default locations.
            env: Environment to load (production/staging/development)
            validate: Whether to validate the config against schema
        Returns:
            Validated configuration dictionary
        Raises:
            ConfigError: If configuration is invalid or missing required sections
        """
        config_file = self._locate_config_file(config_path)
        self.logger.info(f"Loading configuration from {config_file}")
        
        raw_config = self._load_raw_config(config_file)
        config = self._apply_environment_overrides(raw_config, env)
        
        if validate:
            validation = self.validate_config(config)
            if not validation.is_valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(
                    f"{error_type.name}: {msgs}" 
                    for error_type, msgs in validation.errors.items()
                )
                self.logger.error(error_msg)
                raise ConfigError(error_msg, validation)
                
        self._cache = config
        return config
        
    def _locate_config_file(self, config_path: Optional[str]) -> Path:
        """Find configuration file using fallback paths"""
        if config_path:
            candidates = [config_path]
        else:
            candidates = self.DEFAULT_CONFIG_PATHS
            
        for path in candidates:
            try:
                path_obj = Path(path)
                if path_obj.exists():
                    return path_obj
            except Exception as e:
                self.logger.debug(f"Config path check failed for {path}: {str(e)}")
                
        raise FileNotFoundError(
            f"No configuration file found in: {', '.join(str(p) for p in candidates)}"
        )
        
    def _load_raw_config(self, config_file: Path) -> Dict[str, Any]:
        """Load raw config data with format detection"""
        try:
            self._last_modified = os.path.getmtime(config_file)
            
            if config_file.suffix in ('.yaml', '.yml'):
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            elif config_file.suffix == '.json':
                import json
                with open(config_file, 'r') as f:
                    return json.load(f)
            elif config_file.name == '.env':
                from dotenv import dotenv_values
                return dotenv_values(config_file)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        except Exception as e:
            raise ConfigError(
                f"Failed to load config file {config_file}: {str(e)}",
                error_type=ConfigErrorType.INVALID_FORMAT
            ) from e
            
    def _apply_environment_overrides(self, config: Dict[str, Any], env: str) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides"""
        if 'environments' not in config:
            return config
            
        base_config = {k: v for k, v in config.items() if k != 'environments'}
        env_config = config['environments'].get(env, {})
        
        # Deep merge
        from copy import deepcopy
        merged = deepcopy(base_config)
        self._deep_update(merged, env_config)
        
        return merged
        
    def _deep_update(self, original: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively update dictionary"""
        for key, value in update.items():
            if (key in original and isinstance(original[key], dict) 
                    and isinstance(value, dict)):
                self._deep_update(original[key], value)
            else:
                original[key] = value
                
    def validate_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """Comprehensive configuration validation"""
        errors = {err_type: [] for err_type in ConfigErrorType}
        warnings = []
        
        # Check required sections
        for section in self.CONFIG_SCHEMA['required_sections']:
            if section not in config:
                errors[ConfigErrorType.MISSING_SECTION].append(
                    f"Missing required section: {section}"
                )
                
        # Check version
        if 'version' in config:
            if not (self.CONFIG_SCHEMA['version']['min'] <= config['version'] <= 
                    self.CONFIG_SCHEMA['version']['max']):
                errors[ConfigErrorType.VERSION_MISMATCH].append(
                    f"Version {config['version']} not in supported range "
                    f"[{self.CONFIG_SCHEMA['version']['min']}, "
                    f"{self.CONFIG_SCHEMA['version']['max']}]"
                )
                
        # Check value constraints
        for path, constraints in self.CONFIG_SCHEMA['value_constraints'].items():
            keys = path.split('.')
            value = config
            try:
                for key in keys:
                    value = value[key]
                    
                if 'min' in constraints and value < constraints['min']:
                    errors[ConfigErrorType.INVALID_VALUE].append(
                        f"Value too small for {path}: {value} < {constraints['min']}"
                    )
                if 'max' in constraints and value > constraints['max']:
                    errors[ConfigErrorType.INVALID_VALUE].append(
                        f"Value too large for {path}: {value} > {constraints['max']}"
                    )
            except KeyError:
                pass  # Optional values are allowed to be missing
                
        # Filter out empty error categories
        errors = {k: v for k, v in errors.items() if v}
        
        return ConfigValidationResult(
            is_valid=not bool(errors),
            errors=errors,
            warnings=warnings
        )
        
    def check_for_updates(self) -> bool:
        """Check if config file has been modified since last load"""
        if not hasattr(self, '_last_config_path'):
            return False
            
        try:
            current_mtime = os.path.getmtime(self._last_config_path)
            return current_mtime > self._last_modified
        except Exception:
            return False
            
    def reload_if_updated(self) -> Optional[Dict[str, Any]]:
        """Reload config if file has been modified"""
        if self.check_for_updates():
            self.logger.info("Configuration file changed, reloading")
            return self.load(self._last_config_path)
        return None

class ConfigError(Exception):
    """Specialized exception for configuration errors"""
    def __init__(self, message: str, 
                 validation_result: Optional[ConfigValidationResult] = None,
                 error_type: Optional[ConfigErrorType] = None):
        super().__init__(message)
        self.validation_result = validation_result
        self.error_type = error_type
