"""
Data Validator Module - Data Quality and Integrity Assurance
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module provides comprehensive data validation, cleaning, and quality
assurance capabilities for financial data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from enum import Enum
import logging
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationRule(Enum):
    """Types of validation rules."""
    REQUIRED = "required"
    NUMERIC = "numeric"
    POSITIVE = "positive"
    RANGE = "range"
    DATE_FORMAT = "date_format"
    UNIQUE = "unique"
    REGEX = "regex"
    CUSTOM = "custom"
    CONSISTENCY = "consistency"
    OUTLIER = "outlier"


class DataQualityMetric(Enum):
    """Data quality metrics."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    affected_rows: Optional[List[int]] = None
    affected_columns: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "affected_rows": self.affected_rows,
            "affected_columns": self.affected_columns
        }


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    timestamp: datetime
    total_rows: int
    total_columns: int
    validation_results: List[ValidationResult]
    quality_scores: Dict[DataQualityMetric, float]
    summary: Dict[str, Any]
    recommendations: List[str]
    
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        if not self.quality_scores:
            return 0.0
        return np.mean(list(self.quality_scores.values()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "validation_results": [r.to_dict() for r in self.validation_results],
            "quality_scores": {k.value: v for k, v in self.quality_scores.items()},
            "overall_score": self.overall_score(),
            "summary": self.summary,
            "recommendations": self.recommendations
        }


class DataValidator:
    """Main class for data validation."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize data validator.
        
        Args:
            strict_mode: If True, fail on any validation error
        """
        self.strict_mode = strict_mode
        self.validation_rules = {}
        self.custom_validators = {}
        
    def add_rule(self, column: str, rule_type: ValidationRule, 
                params: Optional[Dict[str, Any]] = None):
        """
        Add validation rule for a column.
        
        Args:
            column: Column name or '*' for all columns
            rule_type: Type of validation rule
            params: Rule parameters
        """
        if column not in self.validation_rules:
            self.validation_rules[column] = []
        
        self.validation_rules[column].append({
            'rule_type': rule_type,
            'params': params or {}
        })
    
    def add_custom_validator(self, name: str, validator_func: Callable):
        """
        Add custom validation function.
        
        Args:
            name: Validator name
            validator_func: Function that takes DataFrame and returns ValidationResult
        """
        self.custom_validators[name] = validator_func
    
    def validate(self, data: pd.DataFrame) -> DataQualityReport:
        """
        Validate DataFrame against all rules.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            DataQualityReport with validation results
        """
        validation_results = []
        
        # Apply column-specific rules
        for column, rules in self.validation_rules.items():
            if column == '*':
                # Apply to all columns
                for col in data.columns:
                    for rule in rules:
                        result = self._apply_rule(data, col, rule['rule_type'], rule['params'])
                        validation_results.append(result)
            else:
                if column in data.columns:
                    for rule in rules:
                        result = self._apply_rule(data, column, rule['rule_type'], rule['params'])
                        validation_results.append(result)
                else:
                    validation_results.append(
                        ValidationResult(
                            rule_name=f"column_exists_{column}",
                            passed=False,
                            message=f"Column '{column}' not found in data",
                            severity="error"
                        )
                    )
        
        # Apply custom validators
        for name, validator in self.custom_validators.items():
            try:
                result = validator(data)
                validation_results.append(result)
            except Exception as e:
                validation_results.append(
                    ValidationResult(
                        rule_name=f"custom_{name}",
                        passed=False,
                        message=f"Custom validator '{name}' failed: {str(e)}",
                        severity="error"
                    )
                )
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(data, validation_results)
        
        # Generate summary and recommendations
        summary = self._generate_summary(validation_results)
        recommendations = self._generate_recommendations(data, validation_results, quality_scores)
        
        return DataQualityReport(
            timestamp=datetime.now(),
            total_rows=len(data),
            total_columns=len(data.columns),
            validation_results=validation_results,
            quality_scores=quality_scores,
            summary=summary,
            recommendations=recommendations
        )
    
    def _apply_rule(self, data: pd.DataFrame, column: str, 
                   rule_type: ValidationRule, params: Dict[str, Any]) -> ValidationResult:
        """Apply a validation rule to a column."""
        
        if rule_type == ValidationRule.REQUIRED:
            return self._validate_required(data, column)
        
        elif rule_type == ValidationRule.NUMERIC:
            return self._validate_numeric(data, column)
        
        elif rule_type == ValidationRule.POSITIVE:
            return self._validate_positive(data, column)
        
        elif rule_type == ValidationRule.RANGE:
            return self._validate_range(data, column, params.get('min'), params.get('max'))
        
        elif rule_type == ValidationRule.DATE_FORMAT:
            return self._validate_date_format(data, column, params.get('format'))
        
        elif rule_type == ValidationRule.UNIQUE:
            return self._validate_unique(data, column)
        
        elif rule_type == ValidationRule.REGEX:
            return self._validate_regex(data, column, params.get('pattern'))
        
        elif rule_type == ValidationRule.OUTLIER:
            return self._validate_outliers(data, column, params.get('method', 'iqr'))
        
        else:
            return ValidationResult(
                rule_name=f"unknown_rule_{rule_type.value}",
                passed=False,
                message=f"Unknown validation rule: {rule_type.value}",
                severity="error"
            )
    
    def _validate_required(self, data: pd.DataFrame, column: str) -> ValidationResult:
        """Validate that column has no missing values."""
        missing_count = data[column].isna().sum()
        
        if missing_count == 0:
            return ValidationResult(
                rule_name=f"required_{column}",
                passed=True,
                message=f"Column '{column}' has no missing values"
            )
        else:
            missing_rows = data[data[column].isna()].index.tolist()
            return ValidationResult(
                rule_name=f"required_{column}",
                passed=False,
                message=f"Column '{column}' has {missing_count} missing values",
                severity="error",
                affected_rows=missing_rows[:10],  # First 10 rows
                affected_columns=[column]
            )
    
    def _validate_numeric(self, data: pd.DataFrame, column: str) -> ValidationResult:
        """Validate that column contains numeric values."""
        try:
            pd.to_numeric(data[column], errors='raise')
            return ValidationResult(
                rule_name=f"numeric_{column}",
                passed=True,
                message=f"Column '{column}' contains valid numeric values"
            )
        except:
            non_numeric = data[~data[column].apply(lambda x: isinstance(x, (int, float)) or pd.isna(x))]
            return ValidationResult(
                rule_name=f"numeric_{column}",
                passed=False,
                message=f"Column '{column}' contains non-numeric values",
                severity="error",
                affected_rows=non_numeric.index.tolist()[:10],
                affected_columns=[column]
            )
    
    def _validate_positive(self, data: pd.DataFrame, column: str) -> ValidationResult:
        """Validate that column contains positive values."""
        negative_mask = data[column] < 0
        negative_count = negative_mask.sum()
        
        if negative_count == 0:
            return ValidationResult(
                rule_name=f"positive_{column}",
                passed=True,
                message=f"Column '{column}' contains only positive values"
            )
        else:
            return ValidationResult(
                rule_name=f"positive_{column}",
                passed=False,
                message=f"Column '{column}' has {negative_count} negative values",
                severity="warning",
                affected_rows=data[negative_mask].index.tolist()[:10],
                affected_columns=[column]
            )
    
    def _validate_range(self, data: pd.DataFrame, column: str, 
                       min_val: Optional[float], max_val: Optional[float]) -> ValidationResult:
        """Validate that column values are within range."""
        out_of_range = pd.Series([False] * len(data))
        
        if min_val is not None:
            out_of_range |= (data[column] < min_val)
        
        if max_val is not None:
            out_of_range |= (data[column] > max_val)
        
        out_of_range_count = out_of_range.sum()
        
        if out_of_range_count == 0:
            return ValidationResult(
                rule_name=f"range_{column}",
                passed=True,
                message=f"Column '{column}' values are within range [{min_val}, {max_val}]"
            )
        else:
            return ValidationResult(
                rule_name=f"range_{column}",
                passed=False,
                message=f"Column '{column}' has {out_of_range_count} values out of range [{min_val}, {max_val}]",
                severity="warning",
                affected_rows=data[out_of_range].index.tolist()[:10],
                affected_columns=[column]
            )
    
    def _validate_date_format(self, data: pd.DataFrame, column: str, 
                            date_format: Optional[str]) -> ValidationResult:
        """Validate date format."""
        try:
            if date_format:
                pd.to_datetime(data[column], format=date_format)
            else:
                pd.to_datetime(data[column])
            
            return ValidationResult(
                rule_name=f"date_format_{column}",
                passed=True,
                message=f"Column '{column}' has valid date format"
            )
        except:
            return ValidationResult(
                rule_name=f"date_format_{column}",
                passed=False,
                message=f"Column '{column}' contains invalid date values",
                severity="error",
                affected_columns=[column]
            )
    
    def _validate_unique(self, data: pd.DataFrame, column: str) -> ValidationResult:
        """Validate that column values are unique."""
        duplicates = data[column].duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count == 0:
            return ValidationResult(
                rule_name=f"unique_{column}",
                passed=True,
                message=f"Column '{column}' contains unique values"
            )
        else:
            duplicate_rows = data[duplicates].index.tolist()
            return ValidationResult(
                rule_name=f"unique_{column}",
                passed=False,
                message=f"Column '{column}' has {duplicate_count} duplicate values",
                severity="warning",
                affected_rows=duplicate_rows[:10],
                affected_columns=[column]
            )
    
    def _validate_regex(self, data: pd.DataFrame, column: str, pattern: str) -> ValidationResult:
        """Validate that column matches regex pattern."""
        if not pattern:
            return ValidationResult(
                rule_name=f"regex_{column}",
                passed=False,
                message="No regex pattern provided",
                severity="error"
            )
        
        try:
            matches = data[column].astype(str).str.match(pattern)
            non_matching_count = (~matches).sum()
            
            if non_matching_count == 0:
                return ValidationResult(
                    rule_name=f"regex_{column}",
                    passed=True,
                    message=f"Column '{column}' matches pattern '{pattern}'"
                )
            else:
                non_matching_rows = data[~matches].index.tolist()
                return ValidationResult(
                    rule_name=f"regex_{column}",
                    passed=False,
                    message=f"Column '{column}' has {non_matching_count} values not matching pattern '{pattern}'",
                    severity="warning",
                    affected_rows=non_matching_rows[:10],
                    affected_columns=[column]
                )
        except Exception as e:
            return ValidationResult(
                rule_name=f"regex_{column}",
                passed=False,
                message=f"Regex validation failed: {str(e)}",
                severity="error"
            )
    
    def _validate_outliers(self, data: pd.DataFrame, column: str, method: str) -> ValidationResult:
        """Detect outliers in column."""
        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (data[column] < Q1 - 1.5 * IQR) | (data[column] > Q3 + 1.5 * IQR)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            outliers = z_scores > 3
        
        else:
            return ValidationResult(
                rule_name=f"outliers_{column}",
                passed=False,
                message=f"Unknown outlier detection method: {method}",
                severity="error"
            )
        
        outlier_count = outliers.sum()
        
        if outlier_count == 0:
            return ValidationResult(
                rule_name=f"outliers_{column}",
                passed=True,
                message=f"Column '{column}' has no outliers ({method} method)"
            )
        else:
            outlier_rows = data[outliers].index.tolist()
            return ValidationResult(
                rule_name=f"outliers_{column}",
                passed=False,
                message=f"Column '{column}' has {outlier_count} outliers ({method} method)",
                severity="warning",
                affected_rows=outlier_rows[:10],
                affected_columns=[column]
            )
    
    def _calculate_quality_scores(self, data: pd.DataFrame, 
                                validation_results: List[ValidationResult]) -> Dict[DataQualityMetric, float]:
        """Calculate data quality scores."""
        scores = {}
        
        # Completeness: percentage of non-null values
        total_values = data.shape[0] * data.shape[1]
        non_null_values = data.notna().sum().sum()
        scores[DataQualityMetric.COMPLETENESS] = non_null_values / total_values if total_values > 0 else 0
        
        # Validity: percentage of validation rules passed
        total_rules = len(validation_results)
        passed_rules = sum(1 for r in validation_results if r.passed)
        scores[DataQualityMetric.VALIDITY] = passed_rules / total_rules if total_rules > 0 else 0
        
        # Consistency: check for internal consistency
        consistency_score = 1.0
        # Reduce score for each consistency issue found
        for result in validation_results:
            if 'consistency' in result.rule_name.lower() and not result.passed:
                consistency_score *= 0.9
        scores[DataQualityMetric.CONSISTENCY] = consistency_score
        
        # Accuracy: placeholder - would need ground truth for real calculation
        scores[DataQualityMetric.ACCURACY] = scores[DataQualityMetric.VALIDITY]
        
        # Uniqueness: average uniqueness across columns
        uniqueness_scores = []
        for col in data.columns:
            if data[col].dtype in ['object', 'int64', 'float64']:
                unique_ratio = data[col].nunique() / len(data[col])
                uniqueness_scores.append(unique_ratio)
        scores[DataQualityMetric.UNIQUENESS] = np.mean(uniqueness_scores) if uniqueness_scores else 1.0
        
        # Timeliness: placeholder
        scores[DataQualityMetric.TIMELINESS] = 1.0
        
        return scores
    
    def _generate_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate validation summary."""
        total_checks = len(validation_results)
        passed_checks = sum(1 for r in validation_results if r.passed)
        failed_checks = total_checks - passed_checks
        
        # Count by severity
        severity_counts = {
            'error': sum(1 for r in validation_results if not r.passed and r.severity == 'error'),
            'warning': sum(1 for r in validation_results if not r.passed and r.severity == 'warning'),
            'info': sum(1 for r in validation_results if not r.passed and r.severity == 'info')
        }
        
        # Most common issues
        failed_rules = [r.rule_name for r in validation_results if not r.passed]
        rule_counts = pd.Series(failed_rules).value_counts().to_dict()
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'severity_counts': severity_counts,
            'most_common_issues': dict(list(rule_counts.items())[:5])
        }
    
    def _generate_recommendations(self, data: pd.DataFrame, 
                                validation_results: List[ValidationResult],
                                quality_scores: Dict[DataQualityMetric, float]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for missing data
        missing_data_issues = [r for r in validation_results 
                             if 'required' in r.rule_name and not r.passed]
        if missing_data_issues:
            recommendations.append(
                "Consider imputing missing values or removing rows/columns with excessive missing data"
            )
        
        # Check for outliers
        outlier_issues = [r for r in validation_results 
                        if 'outlier' in r.rule_name and not r.passed]
        if outlier_issues:
            recommendations.append(
                "Review and handle outliers - consider winsorization or removal for extreme values"
            )
        
        # Check data quality scores
        if quality_scores.get(DataQualityMetric.COMPLETENESS, 1) < 0.9:
            recommendations.append(
                "Data completeness is below 90% - investigate sources of missing data"
            )
        
        if quality_scores.get(DataQualityMetric.VALIDITY, 1) < 0.8:
            recommendations.append(
                "Many validation rules are failing - review data sources and collection processes"
            )
        
        # Check for duplicates
        duplicate_issues = [r for r in validation_results 
                          if 'unique' in r.rule_name and not r.passed]
        if duplicate_issues:
            recommendations.append(
                "Duplicate values detected - verify if this is expected or requires deduplication"
            )
        
        # Type issues
        type_issues = [r for r in validation_results 
                     if 'numeric' in r.rule_name or 'date_format' in r.rule_name 
                     and not r.passed]
        if type_issues:
            recommendations.append(
                "Data type inconsistencies found - ensure proper type conversion during data loading"
            )
        
        return recommendations


class DataCleaner:
    """Class for cleaning and preprocessing data."""
    
    def __init__(self):
        """Initialize data cleaner."""
        self.cleaning_log = []
    
    def clean(self, data: pd.DataFrame, 
             validation_report: Optional[DataQualityReport] = None) -> pd.DataFrame:
        """
        Clean data based on validation report.
        
        Args:
            data: DataFrame to clean
            validation_report: Optional validation report to guide cleaning
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Apply cleaning based on validation results
        if validation_report:
            for result in validation_report.validation_results:
                if not result.passed:
                    cleaned_data = self._apply_cleaning(cleaned_data, result)
        
        # Apply general cleaning
        cleaned_data = self._general_cleaning(cleaned_data)
        
        return cleaned_data
    
    def _apply_cleaning(self, data: pd.DataFrame, 
                       validation_result: ValidationResult) -> pd.DataFrame:
        """Apply cleaning based on validation result."""
        
        if 'required' in validation_result.rule_name:
            # Handle missing values
            for col in validation_result.affected_columns or []:
                if col in data.columns:
                    data = self._handle_missing_values(data, col)
        
        elif 'outlier' in validation_result.rule_name:
            # Handle outliers
            for col in validation_result.affected_columns or []:
                if col in data.columns:
                    data = self._handle_outliers(data, col)
        
        elif 'numeric' in validation_result.rule_name:
            # Convert to numeric
            for col in validation_result.affected_columns or []:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
        
        elif 'duplicate' in validation_result.rule_name:
            # Remove duplicates
            data = data.drop_duplicates()
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, column: str, 
                             method: str = 'auto') -> pd.DataFrame:
        """Handle missing values in a column."""
        if method == 'auto':
            # Choose method based on data type and distribution
            if data[column].dtype in ['float64', 'int64']:
                # Use median for numeric data
                fill_value = data[column].median()
                data[column].fillna(fill_value, inplace=True)
                self.cleaning_log.append(f"Filled missing values in '{column}' with median: {fill_value}")
            else:
                # Use mode for categorical data
                fill_value = data[column].mode()[0] if not data[column].mode().empty else 'Unknown'
                data[column].fillna(fill_value, inplace=True)
                self.cleaning_log.append(f"Filled missing values in '{column}' with mode: {fill_value}")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, column: str, 
                        method: str = 'winsorize') -> pd.DataFrame:
        """Handle outliers in a column."""
        if method == 'winsorize':
            # Winsorize at 1st and 99th percentiles
            lower = data[column].quantile(0.01)
            upper = data[column].quantile(0.99)
            data[column] = data[column].clip(lower=lower, upper=upper)
            self.cleaning_log.append(f"Winsorized '{column}' at 1% and 99% percentiles")
        
        elif method == 'remove':
            # Remove outliers using IQR method
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (data[column] < Q1 - 1.5 * IQR) | (data[column] > Q3 + 1.5 * IQR)
            data = data[~outlier_mask]
            self.cleaning_log.append(f"Removed {outlier_mask.sum()} outliers from '{column}'")
        
        return data
    
    def _general_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply general cleaning operations."""
        # Remove completely empty rows and columns
        data = data.dropna(how='all')
        data = data.dropna(axis=1, how='all')
        
        # Strip whitespace from string columns
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].str.strip()
        
        # Sort by index if it's a datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        
        return data


class DataTransformer:
    """Class for data transformations."""
    
    def __init__(self):
        """Initialize data transformer."""
        self.transformations = {}
    
    def add_transformation(self, name: str, transform_func: Callable):
        """Add a transformation function."""
        self.transformations[name] = transform_func
    
    def transform(self, data: pd.DataFrame, 
                 transformations: List[str]) -> pd.DataFrame:
        """Apply transformations to data."""
        transformed_data = data.copy()
        
        for transform_name in transformations:
            if transform_name in self.transformations:
                transform_func = self.transformations[transform_name]
                transformed_data = transform_func(transformed_data)
            else:
                logger.warning(f"Unknown transformation: {transform_name}")
        
        return transformed_data
    
    # Built-in transformations
    def normalize(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize numeric columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    data[col] = (data[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val > 0:
                    data[col] = (data[col] - mean_val) / std_val
        
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for financial data."""
        # Assuming OHLCV data
        if all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            # Simple Moving Averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_window = 20
            bb_std = 2
            data['bb_middle'] = data['close'].rolling(window=bb_window).mean()
            bb_std_dev = data['close'].rolling(window=bb_window).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * bb_std_dev)
            data['bb_lower'] = data['bb_middle'] - (bb_std * bb_std_dev)
        
        return data


def main():
    """Example usage of the data validator module."""
    # Create sample data with some issues
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': dates,
        'price': np.random.normal(100, 10, len(dates)),
        'volume': np.random.randint(1000000, 5000000, len(dates)),
        'returns': np.random.normal(0.001, 0.02, len(dates))
    })
    
    # Introduce some data quality issues
    data.loc[10:20, 'price'] = np.nan  # Missing values
    data.loc[50:60, 'volume'] = -1000  # Negative volumes
    data.loc[100, 'returns'] = 10  # Outlier
    
    # Initialize validator
    validator = DataValidator()
    
    # Add validation rules
    validator.add_rule('price', ValidationRule.REQUIRED)
    validator.add_rule('price', ValidationRule.POSITIVE)
    validator.add_rule('volume', ValidationRule.POSITIVE)
    validator.add_rule('returns', ValidationRule.RANGE, {'min': -0.1, 'max': 0.1})
    validator.add_rule('returns', ValidationRule.OUTLIER, {'method': 'zscore'})
    
    # Validate data
    print("Validating data...")
    report = validator.validate(data)
    
    # Display report
    print(f"\nData Quality Report")
    print(f"Overall Score: {report.overall_score():.2%}")
    print(f"\nQuality Scores:")
    for metric, score in report.quality_scores.items():
        print(f"  {metric.value}: {score:.2%}")
    
    print(f"\nValidation Summary:")
    print(f"  Total checks: {report.summary['total_checks']}")
    print(f"  Passed: {report.summary['passed_checks']}")
    print(f"  Failed: {report.summary['failed_checks']}")
    
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    # Clean data
    print("\nCleaning data...")
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(data, report)
    
    print(f"\nCleaning Log:")
    for log_entry in cleaner.cleaning_log:
        print(f"  - {log_entry}")
    
    # Re-validate cleaned data
    print("\nRe-validating cleaned data...")
    clean_report = validator.validate(cleaned_data)
    print(f"Clean Data Score: {clean_report.overall_score():.2%}")


if __name__ == "__main__":
    main()
