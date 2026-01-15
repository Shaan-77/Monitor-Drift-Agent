"""
Machine learning-based anomaly detection logic.

This module provides functionality to detect anomalies using machine learning
algorithms such as Isolation Forest and Autoencoders.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import os
import pickle

# Try to import sklearn, handle gracefully if not available
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsolationForest = None
    StandardScaler = None

# Import settings for configuration
try:
    from config.settings import get_settings
except ImportError:
    def get_settings():
        return None


class MLAnomalyDetector(ABC):
    """Abstract base class for ML-based anomaly detectors."""
    
    @abstractmethod
    def train(self, training_data: List[Dict]) -> bool:
        """Train the anomaly detection model."""
        pass
    
    @abstractmethod
    def detect(self, data: Dict) -> Dict:
        """Detect anomalies in the given data."""
        pass
    
    @abstractmethod
    def predict(self, data: List[Dict]) -> List[Dict]:
        """Predict anomalies for multiple data points."""
        pass


class IsolationForestDetector(MLAnomalyDetector):
    """Anomaly detector using Isolation Forest algorithm."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies in the data
            n_estimators: Number of trees in the forest
        
        Raises:
            RuntimeError: If sklearn is not available
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is not available. Please install it using: pip install scikit-learn"
            )
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, training_data: List[Dict]) -> bool:
        """
        Train the Isolation Forest model.
        
        Args:
            training_data: List of metric dictionaries for training
        
        Returns:
            True if training successful, False otherwise
        
        Raises:
            ValueError: If training data is insufficient or invalid
            RuntimeError: If sklearn is not available
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not available for model training")
        
        # Validate training data
        if not training_data or len(training_data) == 0:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Training data is empty")
            except ImportError:
                print("Error: Training data is empty")
            return False
        
        # Minimum samples required (at least 2 for meaningful training)
        if len(training_data) < 2:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Insufficient training data: {len(training_data)} samples. Need at least 2.")
            except ImportError:
                print(f"Warning: Insufficient training data: {len(training_data)} samples. Need at least 2.")
            return False
        
        try:
            # Extract and prepare features
            features = self._prepare_features(training_data)
            
            # Fit scaler on training data
            features_scaled = self.scaler.fit_transform(features)
            
            # Train the model
            self.model.fit(features_scaled)
            self.is_trained = True
            
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(f"Successfully trained Isolation Forest model on {len(training_data)} samples")
            except ImportError:
                print(f"Successfully trained Isolation Forest model on {len(training_data)} samples")
            
            return True
        
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error training Isolation Forest model: {e}", exc_info=True)
            except ImportError:
                print(f"Error training Isolation Forest model: {e}")
            self.is_trained = False
            return False
    
    def detect(self, data: Dict) -> Dict:
        """
        Detect anomalies in a single data point.
        
        Args:
            data: Dictionary containing metric values
        
        Returns:
            Dictionary containing anomaly detection result with keys:
            - is_anomaly: bool (True if anomaly detected)
            - anomaly_score: float (lower = more anomalous)
            - prediction: int (-1 for anomaly, 1 for normal)
            - timestamp: str (ISO format)
            - data: original data dictionary
        
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before detection")
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not available for anomaly detection")
        
        try:
            # Extract features from data
            features = self._prepare_features([data])
            
            # Apply same scaler used during training
            features_scaled = self.scaler.transform(features)
            
            # Predict anomaly (returns -1 for anomaly, 1 for normal)
            prediction = self.model.predict(features_scaled)[0]
            
            # Get anomaly score (lower = more anomalous)
            anomaly_score = self.model.score_samples(features_scaled)[0]
            
            return {
                "is_anomaly": bool(prediction == -1),
                "anomaly_score": float(anomaly_score),
                "prediction": int(prediction),
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
        
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error during anomaly detection: {e}", exc_info=True)
            except ImportError:
                print(f"Error during anomaly detection: {e}")
            
            # Return safe default on error
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "prediction": 1,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
    
    def predict(self, data: List[Dict]) -> List[Dict]:
        """
        Predict anomalies for multiple data points (batch prediction).
        
        Args:
            data: List of metric dictionaries
        
        Returns:
            List of anomaly detection results
        
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not available for batch prediction")
        
        try:
            # Prepare features for all data points
            features = self._prepare_features(data)
            
            # Apply scaler
            features_scaled = self.scaler.transform(features)
            
            # Batch predict
            predictions = self.model.predict(features_scaled)
            anomaly_scores = self.model.score_samples(features_scaled)
            
            # Format results
            results = []
            for i, point in enumerate(data):
                results.append({
                    "is_anomaly": bool(predictions[i] == -1),
                    "anomaly_score": float(anomaly_scores[i]),
                    "prediction": int(predictions[i]),
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": point
                })
            
            return results
        
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error during batch prediction: {e}", exc_info=True)
            except ImportError:
                print(f"Error during batch prediction: {e}")
            
            # Fallback to individual detection
            results = []
            for point in data:
                results.append(self.detect(point))
            return results
    
    def save_model(self, model_path: Optional[str] = None) -> bool:
        """
        Save the trained model and scaler to disk.
        
        Args:
            model_path: Optional path to save model. If None, uses settings.ml_model_path
        
        Returns:
            True if save successful, False otherwise
        """
        if not self.is_trained:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning("Cannot save untrained model")
            except ImportError:
                print("Warning: Cannot save untrained model")
            return False
        
        # Get model path from settings if not provided
        if model_path is None:
            settings = get_settings()
            if settings:
                model_path = settings.ml_model_path
            else:
                model_path = "models/isolation_forest_model.pkl"
        
        try:
            # Create directory if it doesn't exist
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            
            # Prepare data to save: model, scaler, and metadata
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'is_trained': self.is_trained
            }
            
            # Save using pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(f"Successfully saved model to {model_path}")
            except ImportError:
                print(f"Successfully saved model to {model_path}")
            
            return True
        
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error saving model to {model_path}: {e}", exc_info=True)
            except ImportError:
                print(f"Error saving model to {model_path}: {e}")
            return False
    
    @classmethod
    def load_model(cls, model_path: Optional[str] = None) -> Optional['IsolationForestDetector']:
        """
        Load a trained model and scaler from disk.
        
        Args:
            model_path: Optional path to load model from. If None, uses settings.ml_model_path
        
        Returns:
            IsolationForestDetector instance with loaded model, or None if loading failed
        """
        if not SKLEARN_AVAILABLE:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("scikit-learn is not available. Cannot load model.")
            except ImportError:
                print("Error: scikit-learn is not available. Cannot load model.")
            return None
        
        # Get model path from settings if not provided
        if model_path is None:
            settings = get_settings()
            if settings:
                model_path = settings.ml_model_path
            else:
                model_path = "models/isolation_forest_model.pkl"
        
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Model file not found: {model_path}")
                except ImportError:
                    print(f"Warning: Model file not found: {model_path}")
                return None
            
            # Load model data
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create detector instance
            detector = cls(
                contamination=model_data.get('contamination', 0.1),
                n_estimators=model_data.get('n_estimators', 100)
            )
            
            # Restore model state
            detector.model = model_data['model']
            detector.scaler = model_data['scaler']
            detector.is_trained = model_data.get('is_trained', True)
            
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(f"Successfully loaded model from {model_path}")
            except ImportError:
                print(f"Successfully loaded model from {model_path}")
            
            return detector
        
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
            except ImportError:
                print(f"Error loading model from {model_path}: {e}")
            return None
    
    def _prepare_features(self, data: List[Dict]) -> np.ndarray:
        """
        Prepare feature vectors from metric data.
        
        Extracts cost values from training data dictionaries and normalizes them.
        Supports both single-feature (cost only) and multi-feature scenarios.
        
        Args:
            data: List of metric dictionaries with 'cost' key
        
        Returns:
            Numpy array of feature vectors with shape (n_samples, n_features)
        
        Raises:
            ValueError: If data is empty or missing required keys
        """
        if not data or len(data) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Extract cost values from dictionaries
        features = []
        for item in data:
            if isinstance(item, dict):
                # Extract cost value
                if 'cost' in item:
                    cost_value = float(item['cost'])
                    features.append([cost_value])
                else:
                    raise ValueError(f"Missing 'cost' key in data item: {item}")
            elif isinstance(item, (int, float)):
                # Direct numeric value
                features.append([float(item)])
            else:
                raise ValueError(f"Invalid data type in training data: {type(item)}")
        
        # Convert to numpy array
        features_array = np.array(features)
        
        # Reshape if single feature (ensure 2D array)
        if features_array.ndim == 1:
            features_array = features_array.reshape(-1, 1)
        
        return features_array


class AutoencoderDetector(MLAnomalyDetector):
    """Anomaly detector using Autoencoder neural network."""
    
    def __init__(self, encoding_dim: int = 32, epochs: int = 50):
        """
        Initialize Autoencoder detector.
        
        Args:
            encoding_dim: Dimension of the encoding layer
            epochs: Number of training epochs
        
        TODO: Initialize Autoencoder model using TensorFlow/Keras
        """
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.model = None
        # TODO: Initialize Autoencoder model
        # from tensorflow import keras
        # self.model = self._build_model()
    
    def train(self, training_data: List[Dict]) -> bool:
        """
        Train the Autoencoder model.
        
        Args:
            training_data: List of metric dictionaries for training
        
        Returns:
            True if training successful, False otherwise
        
        TODO: Implement model training
        """
        # TODO: Prepare training data and train autoencoder
        # X = self._prepare_features(training_data)
        # self.model.fit(X, X, epochs=self.epochs, verbose=0)
        return False
    
    def detect(self, data: Dict) -> Dict:
        """
        Detect anomalies in a single data point.
        
        Args:
            data: Dictionary containing metric values
        
        Returns:
            Dictionary containing anomaly detection result
        
        TODO: Implement anomaly detection using reconstruction error
        """
        # TODO: Calculate reconstruction error and determine if anomaly
        # features = self._prepare_features([data])
        # reconstructed = self.model.predict(features)
        # reconstruction_error = np.mean(np.square(features - reconstructed))
        # threshold = 0.1  # TODO: Determine threshold from training data
        
        return {
            "is_anomaly": False,
            "reconstruction_error": 0.0,
            "threshold": 0.0,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    
    def predict(self, data: List[Dict]) -> List[Dict]:
        """
        Predict anomalies for multiple data points.
        
        Args:
            data: List of metric dictionaries
        
        Returns:
            List of anomaly detection results
        
        TODO: Implement batch prediction
        """
        # TODO: Implement batch prediction
        results = []
        for point in data:
            results.append(self.detect(point))
        return results
    
    def _prepare_features(self, data: List[Dict]) -> np.ndarray:
        """
        Prepare feature vectors from metric data.
        
        Args:
            data: List of metric dictionaries
        
        Returns:
            Numpy array of feature vectors
        
        TODO: Implement feature extraction and normalization
        """
        # TODO: Extract and normalize features from metric data
        return np.array([])
    
    def _build_model(self):
        """
        Build the Autoencoder model architecture.
        
        Returns:
            Compiled Keras model
        
        TODO: Implement model architecture
        """
        # TODO: Implement encoder-decoder architecture
        return None


def get_historical_training_data(
    resource_name: Optional[str] = None,
    days: Optional[int] = None
) -> List[Dict]:
    """
    Fetch historical cost data from PostgreSQL for model training.
    
    This function retrieves historical cloud resource cost data from the database
    and converts it to a format suitable for training the ML model.
    
    Args:
        resource_name: Optional resource name to fetch data for specific resource.
                      If None, fetches data for all resources (aggregated).
        days: Number of days to look back (defaults to settings.ml_training_lookback_days)
    
    Returns:
        List of dictionaries with 'cost' key, suitable for training.
        Format: [{"cost": float}, {"cost": float}, ...]
    
    Raises:
        RuntimeError: If database module is not available
    """
    try:
        from data_collection.database import get_historical_cost_data, connect_to_db
        from psycopg2 import Error
    except ImportError:
        raise RuntimeError(
            "Database module not available. Cannot fetch historical training data."
        )
    
    # Get settings for default days
    settings = get_settings()
    if days is None:
        if settings:
            days = settings.ml_training_lookback_days
        else:
            days = 30  # Default fallback
    
    training_data = []
    
    try:
        if resource_name:
            # Fetch data for specific resource
            cost_values = get_historical_cost_data(resource_name, days)
            training_data = [{"cost": float(cost)} for cost in cost_values if cost is not None]
        else:
            # Fetch data for all resources (aggregate)
            conn = connect_to_db()
            if conn is None:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error("Failed to connect to database for training data")
                except ImportError:
                    print("Error: Failed to connect to database for training data")
                return []
            
            try:
                cursor = conn.cursor()
                from datetime import timedelta
                cutoff_timestamp = datetime.utcnow() - timedelta(days=days)
                
                # Query all cost data within the time period
                query = """
                    SELECT cost FROM cloud_resource_costs
                    WHERE timestamp >= %s
                    ORDER BY timestamp DESC;
                """
                
                cursor.execute(query, (cutoff_timestamp,))
                results = cursor.fetchall()
                
                # Extract cost values
                training_data = [
                    {"cost": float(row[0])}
                    for row in results
                    if row[0] is not None
                ]
                
                cursor.close()
                conn.close()
            
            except Error as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.error(f"Database error fetching training data: {e}")
                except ImportError:
                    print(f"Error fetching training data from database: {e}")
                if conn:
                    conn.close()
                return []
        
        if len(training_data) == 0:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"No historical cost data found for training (resource: {resource_name}, days: {days})")
            except ImportError:
                print(f"Warning: No historical cost data found for training (resource: {resource_name}, days: {days})")
        
        return training_data
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error fetching historical training data: {e}", exc_info=True)
        except ImportError:
            print(f"Error fetching historical training data: {e}")
        return []


def train_model_from_database(
    resource_name: Optional[str] = None,
    contamination: Optional[float] = None,
    n_estimators: Optional[int] = None,
    days: Optional[int] = None
) -> Optional[IsolationForestDetector]:
    """
    Train Isolation Forest model using historical data from database.
    
    This function orchestrates the complete training process:
    1. Fetches historical cost data from PostgreSQL
    2. Creates IsolationForestDetector instance
    3. Trains the model
    4. Saves the model to disk
    
    Args:
        resource_name: Optional resource name to train model for specific resource.
                      If None, trains on aggregated data from all resources.
        contamination: Expected proportion of anomalies (defaults to settings.ml_contamination)
        n_estimators: Number of trees (defaults to settings.ml_n_estimators)
        days: Number of days to look back (defaults to settings.ml_training_lookback_days)
    
    Returns:
        Trained IsolationForestDetector instance if successful, None otherwise
    """
    if not SKLEARN_AVAILABLE:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("scikit-learn is not available. Cannot train model.")
        except ImportError:
            print("Error: scikit-learn is not available. Cannot train model.")
        return None
    
    # Get settings
    settings = get_settings()
    if settings is None:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Settings not available. Cannot determine model parameters.")
        except ImportError:
            print("Error: Settings not available. Cannot determine model parameters.")
        return None
    
    # Use settings defaults if not provided
    if contamination is None:
        contamination = settings.ml_contamination
    if n_estimators is None:
        n_estimators = settings.ml_n_estimators
    if days is None:
        days = settings.ml_training_lookback_days
    
    try:
        # Fetch historical training data
        training_data = get_historical_training_data(resource_name, days)
        
        if len(training_data) < 2:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Insufficient training data: {len(training_data)} samples. Need at least 2.")
            except ImportError:
                print(f"Warning: Insufficient training data: {len(training_data)} samples. Need at least 2.")
            return None
        
        # Create detector instance
        detector = IsolationForestDetector(
            contamination=contamination,
            n_estimators=n_estimators
        )
        
        # Train the model
        success = detector.train(training_data)
        
        if not success:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Model training failed")
            except ImportError:
                print("Error: Model training failed")
            return None
        
        # Save the model
        try:
            detector.save_model()
        except Exception as e:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Failed to save model: {e}. Model is still trained in memory.")
            except ImportError:
                print(f"Warning: Failed to save model: {e}. Model is still trained in memory.")
        
        return detector
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error training model from database: {e}", exc_info=True)
        except ImportError:
            print(f"Error training model from database: {e}")
        return None


def detect_anomaly(
    model: IsolationForestDetector,
    real_time_data: Dict | List[Dict]
) -> Dict | List[Dict]:
    """
    Detect anomalies in real-time data using a trained model.
    
    This is a standalone function that handles both single point and batch detection.
    
    Args:
        model: Trained IsolationForestDetector instance
        real_time_data: Single dictionary or list of dictionaries with 'cost' key
    
    Returns:
        Single detection result dictionary or list of detection results
    
    Raises:
        RuntimeError: If model is not trained
        ValueError: If data format is invalid
    """
    if not model.is_trained:
        raise RuntimeError("Model must be trained before detection")
    
    # Handle single data point
    if isinstance(real_time_data, dict):
        return model.detect(real_time_data)
    
    # Handle batch data
    elif isinstance(real_time_data, list):
        if len(real_time_data) == 0:
            raise ValueError("Real-time data list cannot be empty")
        return model.predict(real_time_data)
    
    else:
        raise ValueError(f"Invalid data type: {type(real_time_data)}. Expected dict or list of dicts.")


def detect_anomaly_from_realtime_cost(
    resource_name: Optional[str] = None,
    model: Optional[IsolationForestDetector] = None
) -> Optional[Dict]:
    """
    Fetch current cost data and detect anomalies.
    
    This function fetches the latest cost data from the database (or from cloud metrics)
    and runs anomaly detection on it.
    
    Args:
        resource_name: Optional resource name to check. If None, checks all resources.
        model: Optional trained model instance. If None, loads from disk.
    
    Returns:
        Detection result dictionary, or None if detection failed
    """
    try:
        from data_collection.database import connect_to_db
        from psycopg2 import Error
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Database module not available for real-time cost detection")
        except ImportError:
            print("Error: Database module not available for real-time cost detection")
        return None
    
    # Load model if not provided
    if model is None:
        model = IsolationForestDetector.load_model()
        if model is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to load trained model for anomaly detection")
            except ImportError:
                print("Error: Failed to load trained model for anomaly detection")
            return None
    
    try:
        # Fetch latest cost data from database
        conn = connect_to_db()
        if conn is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to connect to database for real-time cost data")
            except ImportError:
                print("Error: Failed to connect to database for real-time cost data")
            return None
        
        cursor = conn.cursor()
        
        if resource_name:
            # Fetch latest cost for specific resource
            query = """
                SELECT resource_name, cost, timestamp
                FROM cloud_resource_costs
                WHERE resource_name = %s
                ORDER BY timestamp DESC
                LIMIT 1;
            """
            cursor.execute(query, (resource_name,))
        else:
            # Fetch latest cost for any resource (most recent)
            query = """
                SELECT resource_name, cost, timestamp
                FROM cloud_resource_costs
                ORDER BY timestamp DESC
                LIMIT 1;
            """
            cursor.execute(query)
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"No recent cost data found (resource: {resource_name})")
            except ImportError:
                print(f"Warning: No recent cost data found (resource: {resource_name})")
            return None
        
        # Format data for detection
        data = {
            "cost": float(result[1]),
            "resource_name": result[0],
            "timestamp": result[2]
        }
        
        # Run detection
        detection_result = detect_anomaly(model, data)
        
        # Add resource context to result
        if isinstance(detection_result, dict):
            detection_result['resource_name'] = data['resource_name']
        
        return detection_result
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error fetching real-time cost: {e}")
        except ImportError:
            print(f"Error fetching real-time cost from database: {e}")
        return None
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error detecting anomaly from real-time cost: {e}", exc_info=True)
        except ImportError:
            print(f"Error detecting anomaly from real-time cost: {e}")
        return None


def detect_anomalies_for_all_resources(
    model: Optional[IsolationForestDetector] = None
) -> List[Dict]:
    """
    Detect anomalies for all resources in the database.
    
    Fetches the latest cost data for each distinct resource and runs anomaly detection.
    
    Args:
        model: Optional trained model instance. If None, loads from disk.
    
    Returns:
        List of detection results, one per resource
    """
    try:
        from data_collection.database import connect_to_db
        from psycopg2 import Error
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Database module not available for multi-resource detection")
        except ImportError:
            print("Error: Database module not available for multi-resource detection")
        return []
    
    # Load model if not provided
    if model is None:
        model = IsolationForestDetector.load_model()
        if model is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to load trained model for multi-resource detection")
            except ImportError:
                print("Error: Failed to load trained model for multi-resource detection")
            return []
    
    results = []
    
    try:
        conn = connect_to_db()
        if conn is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to connect to database for multi-resource detection")
            except ImportError:
                print("Error: Failed to connect to database for multi-resource detection")
            return []
        
        cursor = conn.cursor()
        
        # Get distinct resource names
        query = """
            SELECT DISTINCT resource_name
            FROM cloud_resource_costs
            ORDER BY resource_name;
        """
        cursor.execute(query)
        resource_names = [row[0] for row in cursor.fetchall()]
        
        # Detect anomalies for each resource
        for resource_name in resource_names:
            detection_result = detect_anomaly_from_realtime_cost(resource_name, model)
            if detection_result:
                results.append(detection_result)
        
        cursor.close()
        conn.close()
        
        return results
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error in multi-resource detection: {e}")
        except ImportError:
            print(f"Error in multi-resource detection: {e}")
        return []
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error in multi-resource anomaly detection: {e}", exc_info=True)
        except ImportError:
            print(f"Error in multi-resource anomaly detection: {e}")
        return []


def trigger_alert_if_anomaly(
    detection_result: Dict,
    anomaly_score_threshold: Optional[float] = None
) -> bool:
    """
    Trigger an alert if anomaly is detected and meets confidence threshold.
    
    This function checks if an anomaly was detected and if the anomaly score
    is below the threshold (indicating high confidence). If both conditions
    are met, it triggers an alert using the existing alert_trigger module.
    
    Args:
        detection_result: Detection result dictionary from detect_anomaly()
        anomaly_score_threshold: Optional threshold for anomaly score.
                                If None, uses settings.ml_anomaly_score_threshold.
                                Only alerts if anomaly_score < threshold.
    
    Returns:
        True if alert was triggered, False otherwise
    """
    # Check if anomaly was detected
    if not detection_result.get('is_anomaly', False):
        return False
    
    # Get threshold from settings if not provided
    if anomaly_score_threshold is None:
        settings = get_settings()
        if settings:
            anomaly_score_threshold = settings.ml_anomaly_score_threshold
        else:
            anomaly_score_threshold = -0.5  # Default fallback
    
    # Check if anomaly score meets threshold (lower = more anomalous)
    anomaly_score = detection_result.get('anomaly_score', 0.0)
    if anomaly_score >= anomaly_score_threshold:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(
                f"Anomaly detected but score ({anomaly_score:.3f}) above threshold "
                f"({anomaly_score_threshold:.3f}). Not alerting to reduce false positives."
            )
        except ImportError:
            print(
                f"Anomaly detected but score ({anomaly_score:.3f}) above threshold "
                f"({anomaly_score_threshold:.3f}). Not alerting."
            )
        return False
    
    # Extract resource information from detection result
    data = detection_result.get('data', {})
    cost_value = data.get('cost', 0.0)
    resource_name = data.get('resource_name', detection_result.get('resource_name', 'Unknown'))
    
    # Extract provider from resource_name for resource_type
    resource_type = "Cloud"
    if isinstance(resource_name, str):
        if resource_name.startswith("AWS"):
            resource_type = "AWS"
        elif resource_name.startswith("GCP"):
            resource_type = "GCP"
        elif resource_name.startswith("Azure"):
            resource_type = "Azure"
    
    # Get timestamp
    timestamp_str = detection_result.get('timestamp')
    if timestamp_str:
        try:
            from datetime import datetime
            # Parse ISO format timestamp
            if 'T' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                # Convert to UTC naive if timezone-aware
                if timestamp.tzinfo:
                    timestamp = timestamp.utctimetuple()
                    timestamp = datetime(*timestamp[:6])
            else:
                timestamp = datetime.utcnow()
        except Exception:
            timestamp = datetime.utcnow()
    else:
        timestamp = datetime.utcnow()
    
    # Trigger alert
    try:
        from anomaly_detection.alert_trigger import trigger_alert
        
        metric_name = f"Cloud Cost (ML Anomaly) - Score: {anomaly_score:.3f}"
        alert_triggered = trigger_alert(
            metric_name=metric_name,
            value=float(cost_value),
            timestamp=timestamp,
            resource_type=resource_type
        )
        
        if alert_triggered:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(
                    f"ML anomaly alert triggered for {resource_name}: "
                    f"cost={cost_value}, score={anomaly_score:.3f}"
                )
            except ImportError:
                print(
                    f"ML anomaly alert triggered for {resource_name}: "
                    f"cost={cost_value}, score={anomaly_score:.3f}"
                )
        
        return alert_triggered
    
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Alert trigger module not available")
        except ImportError:
            print("Error: Alert trigger module not available")
        return False
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error triggering ML anomaly alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error triggering ML anomaly alert: {e}")
        return False


def detect_and_alert(
    resource_name: Optional[str] = None,
    model: Optional[IsolationForestDetector] = None,
    check_all_resources: bool = False
) -> Dict[str, Any]:
    """
    Orchestrate complete flow: detection → alert triggering.
    
    This function:
    1. Loads or uses provided model
    2. Fetches real-time cost data
    3. Runs anomaly detection
    4. Triggers alerts if anomalies detected
    5. Returns summary of detections and alerts
    
    Args:
        resource_name: Optional resource name to check. If None and check_all_resources=False,
                      checks most recent resource.
        model: Optional trained model instance. If None, loads from disk.
        check_all_resources: If True, checks all resources. Overrides resource_name.
    
    Returns:
        Dictionary with summary:
        {
            'detections': List[Dict],  # All detection results
            'alerts_triggered': int,    # Number of alerts triggered
            'anomalies_found': int      # Number of anomalies detected
        }
    """
    summary = {
        'detections': [],
        'alerts_triggered': 0,
        'anomalies_found': 0
    }
    
    # Load model if not provided
    if model is None:
        model = IsolationForestDetector.load_model()
        if model is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to load trained model for detect_and_alert")
            except ImportError:
                print("Error: Failed to load trained model for detect_and_alert")
            return summary
    
    try:
        if check_all_resources:
            # Check all resources
            detection_results = detect_anomalies_for_all_resources(model)
        else:
            # Check single resource
            detection_result = detect_anomaly_from_realtime_cost(resource_name, model)
            if detection_result:
                detection_results = [detection_result]
            else:
                detection_results = []
        
        summary['detections'] = detection_results
        
        # Process each detection result
        for detection_result in detection_results:
            # Count anomalies
            if detection_result.get('is_anomaly', False):
                summary['anomalies_found'] += 1
                
                # Try to trigger alert
                alert_triggered = trigger_alert_if_anomaly(detection_result)
                if alert_triggered:
                    summary['alerts_triggered'] += 1
        
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(
                f"ML detection complete: {summary['anomalies_found']} anomalies found, "
                f"{summary['alerts_triggered']} alerts triggered"
            )
        except ImportError:
            print(
                f"ML detection complete: {summary['anomalies_found']} anomalies found, "
                f"{summary['alerts_triggered']} alerts triggered"
            )
        
        # Track performance if enabled
        if track_performance:
            settings = get_settings()
            if settings and settings.ml_enabled:
                try:
                    # Calculate evaluation period (current time window)
                    evaluation_period_end = datetime.utcnow()
                    # Use a default 1-hour window if start not specified
                    from datetime import timedelta
                    evaluation_period_start = evaluation_period_end - timedelta(hours=1)
                    
                    # Store performance metrics
                    store_performance_from_detection_summary(
                        summary,
                        evaluation_period_start,
                        evaluation_period_end
                    )
                except Exception as e:
                    try:
                        from utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.warning(f"Failed to store performance metrics: {e}")
                    except ImportError:
                        print(f"Warning: Failed to store performance metrics: {e}")
        
        return summary
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error in detect_and_alert: {e}", exc_info=True)
        except ImportError:
            print(f"Error in detect_and_alert: {e}")
        return summary


def calculate_performance_metrics(
    detection_results: List[Dict],
    alerts_triggered: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate performance metrics from detection results.
    
    This function computes various performance metrics for the ML model
    based on detection results. Since Isolation Forest is unsupervised,
    we track detection rate, alert rate, and average anomaly scores.
    
    Args:
        detection_results: List of detection result dictionaries from detect() or predict()
        alerts_triggered: Optional count of alerts triggered (if not in detection_results)
    
    Returns:
        Dictionary containing:
        {
            'total_predictions': int,
            'anomalies_detected': int,
            'alerts_triggered': int,
            'avg_anomaly_score': float,
            'detection_rate': float,
            'alert_rate': float,
            'false_positive_rate': Optional[float]  # None if not available
        }
    """
    if not detection_results or len(detection_results) == 0:
        return {
            'total_predictions': 0,
            'anomalies_detected': 0,
            'alerts_triggered': 0,
            'avg_anomaly_score': 0.0,
            'detection_rate': 0.0,
            'alert_rate': 0.0,
            'false_positive_rate': None
        }
    
    total_predictions = len(detection_results)
    anomalies_detected = sum(1 for result in detection_results if result.get('is_anomaly', False))
    
    # Extract anomaly scores
    anomaly_scores = [
        result.get('anomaly_score', 0.0)
        for result in detection_results
        if 'anomaly_score' in result
    ]
    
    # Calculate average anomaly score
    avg_anomaly_score = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
    
    # Calculate detection rate
    detection_rate = float(anomalies_detected / total_predictions) if total_predictions > 0 else 0.0
    
    # Get alerts triggered count
    if alerts_triggered is not None:
        alerts_count = alerts_triggered
    else:
        # Try to extract from detection results if available
        alerts_count = sum(1 for result in detection_results if result.get('alert_triggered', False))
    
    # Calculate alert rate (alerts_triggered / anomalies_detected)
    if anomalies_detected > 0:
        alert_rate = float(alerts_count / anomalies_detected)
    else:
        alert_rate = 0.0
    
    return {
        'total_predictions': total_predictions,
        'anomalies_detected': anomalies_detected,
        'alerts_triggered': alerts_count,
        'avg_anomaly_score': avg_anomaly_score,
        'detection_rate': detection_rate,
        'alert_rate': alert_rate,
        'false_positive_rate': None  # Requires labeled data or feedback mechanism
    }


def store_model_performance(
    metrics: Dict[str, Any],
    evaluation_period_start: Optional[datetime] = None,
    evaluation_period_end: Optional[datetime] = None,
    model_version: Optional[str] = None
) -> bool:
    """
    Store model performance metrics in PostgreSQL database.
    
    This function stores calculated performance metrics in the model_performance
    table for later analysis and retraining decisions.
    
    Args:
        metrics: Dictionary from calculate_performance_metrics() with performance metrics
        evaluation_period_start: Optional start timestamp of evaluation period
        evaluation_period_end: Optional end timestamp of evaluation period
        model_version: Optional model version identifier
    
    Returns:
        True if storage successful, False otherwise
    """
    try:
        from data_collection.database import connect_to_db
        from psycopg2 import Error
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Database module not available for storing model performance")
        except ImportError:
            print("Error: Database module not available for storing model performance")
        return False
    
    # Validate required metrics
    required_keys = ['total_predictions', 'anomalies_detected', 'alerts_triggered', 'detection_rate']
    if not all(key in metrics for key in required_keys):
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Missing required metrics: {set(required_keys) - set(metrics.keys())}")
        except ImportError:
            print(f"Error: Missing required metrics: {set(required_keys) - set(metrics.keys())}")
        return False
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to connect to database for storing model performance")
            except ImportError:
                print("Error: Failed to connect to database for storing model performance")
            return False
        
        cursor = conn.cursor()
        
        # Prepare timestamp values (convert to UTC naive if timezone-aware)
        period_start = evaluation_period_start
        if period_start and period_start.tzinfo:
            period_start = period_start.replace(tzinfo=None)
        
        period_end = evaluation_period_end
        if period_end and period_end.tzinfo:
            period_end = period_end.replace(tzinfo=None)
        
        # Insert performance metrics
        insert_query = """
            INSERT INTO model_performance (
                model_version, total_predictions, anomalies_detected, alerts_triggered,
                avg_anomaly_score, detection_rate, alert_rate, false_positive_rate,
                evaluation_period_start, evaluation_period_end, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        cursor.execute(
            insert_query,
            (
                model_version,
                metrics['total_predictions'],
                metrics['anomalies_detected'],
                metrics['alerts_triggered'],
                metrics.get('avg_anomaly_score'),
                metrics['detection_rate'],
                metrics.get('alert_rate'),
                metrics.get('false_positive_rate'),
                period_start,
                period_end,
                datetime.utcnow()
            )
        )
        
        conn.commit()
        cursor.close()
        
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(
                f"Successfully stored model performance: "
                f"{metrics['total_predictions']} predictions, "
                f"{metrics['anomalies_detected']} anomalies, "
                f"detection_rate={metrics['detection_rate']:.3f}"
            )
        except ImportError:
            print(
                f"Successfully stored model performance: "
                f"{metrics['total_predictions']} predictions, "
                f"{metrics['anomalies_detected']} anomalies"
            )
        
        return True
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error storing model performance: {e}")
        except ImportError:
            print(f"Error storing model performance in database: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error storing model performance: {e}", exc_info=True)
        except ImportError:
            print(f"Error storing model performance: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def store_performance_from_detection_summary(
    detection_summary: Dict[str, Any],
    evaluation_period_start: Optional[datetime] = None,
    evaluation_period_end: Optional[datetime] = None
) -> bool:
    """
    Store performance metrics from detect_and_alert() summary.
    
    This is a convenience wrapper that extracts metrics from the detection summary
    and stores them in the database.
    
    Args:
        detection_summary: Summary dictionary from detect_and_alert() with keys:
                          'detections', 'alerts_triggered', 'anomalies_found'
        evaluation_period_start: Optional start timestamp of evaluation period
        evaluation_period_end: Optional end timestamp of evaluation period
    
    Returns:
        True if storage successful, False otherwise
    """
    detections = detection_summary.get('detections', [])
    alerts_triggered = detection_summary.get('alerts_triggered', 0)
    
    # Calculate metrics from detection results
    metrics = calculate_performance_metrics(detections, alerts_triggered)
    
    # Store in database
    return store_model_performance(metrics, evaluation_period_start, evaluation_period_end)


def fetch_model_performance_data(
    days: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fetch recent model performance data from database.
    
    This function retrieves performance records from the model_performance table
    and calculates aggregate metrics over the specified time period.
    
    Args:
        days: Number of days to look back (defaults to settings.ml_performance_lookback_days)
    
    Returns:
        Dictionary containing:
        {
            'records': List[Dict],  # Raw performance records
            'aggregate_metrics': Dict,  # Aggregated metrics
            'total_records': int
        }
    """
    try:
        from data_collection.database import connect_to_db
        from psycopg2 import Error
        from datetime import timedelta
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Database module not available for fetching performance data")
        except ImportError:
            print("Error: Database module not available for fetching performance data")
        return {'records': [], 'aggregate_metrics': {}, 'total_records': 0}
    
    # Get days from settings if not provided
    if days is None:
        settings = get_settings()
        if settings:
            days = settings.ml_performance_lookback_days
        else:
            days = 7  # Default fallback
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Failed to connect to database for fetching performance data")
            except ImportError:
                print("Error: Failed to connect to database for fetching performance data")
            return {'records': [], 'aggregate_metrics': {}, 'total_records': 0}
        
        cursor = conn.cursor()
        
        # Calculate cutoff timestamp
        cutoff_timestamp = datetime.utcnow() - timedelta(days=days)
        
        # Query performance records
        query = """
            SELECT 
                id, model_version, total_predictions, anomalies_detected,
                alerts_triggered, avg_anomaly_score, detection_rate,
                alert_rate, false_positive_rate, evaluation_period_start,
                evaluation_period_end, timestamp
            FROM model_performance
            WHERE timestamp >= %s
            ORDER BY timestamp DESC;
        """
        
        cursor.execute(query, (cutoff_timestamp,))
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        records = []
        for row in results:
            records.append({
                'id': row[0],
                'model_version': row[1],
                'total_predictions': row[2],
                'anomalies_detected': row[3],
                'alerts_triggered': row[4],
                'avg_anomaly_score': float(row[5]) if row[5] is not None else None,
                'detection_rate': float(row[6]) if row[6] is not None else 0.0,
                'alert_rate': float(row[7]) if row[7] is not None else None,
                'false_positive_rate': float(row[8]) if row[8] is not None else None,
                'evaluation_period_start': row[9],
                'evaluation_period_end': row[10],
                'timestamp': row[11]
            })
        
        cursor.close()
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        if records:
            total_predictions = sum(r['total_predictions'] for r in records)
            total_anomalies = sum(r['anomalies_detected'] for r in records)
            total_alerts = sum(r['alerts_triggered'] for r in records)
            
            # Average detection rate
            detection_rates = [r['detection_rate'] for r in records if r['detection_rate'] is not None]
            avg_detection_rate = float(np.mean(detection_rates)) if detection_rates else 0.0
            
            # Average alert rate
            alert_rates = [r['alert_rate'] for r in records if r['alert_rate'] is not None]
            avg_alert_rate = float(np.mean(alert_rates)) if alert_rates else 0.0
            
            # Average anomaly score
            anomaly_scores = [r['avg_anomaly_score'] for r in records if r['avg_anomaly_score'] is not None]
            avg_anomaly_score = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
            
            aggregate_metrics = {
                'total_predictions': total_predictions,
                'total_anomalies_detected': total_anomalies,
                'total_alerts_triggered': total_alerts,
                'avg_detection_rate': avg_detection_rate,
                'avg_alert_rate': avg_alert_rate,
                'avg_anomaly_score': avg_anomaly_score,
                'num_records': len(records)
            }
        
        return {
            'records': records,
            'aggregate_metrics': aggregate_metrics,
            'total_records': len(records)
        }
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error fetching performance data: {e}")
        except ImportError:
            print(f"Error fetching performance data from database: {e}")
        return {'records': [], 'aggregate_metrics': {}, 'total_records': 0}
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error fetching model performance data: {e}", exc_info=True)
        except ImportError:
            print(f"Error fetching model performance data: {e}")
        return {'records': [], 'aggregate_metrics': {}, 'total_records': 0}
    finally:
        if conn:
            conn.close()


def get_latest_performance_metrics() -> Optional[Dict[str, Any]]:
    """
    Get the most recent model performance record.
    
    Returns:
        Dictionary with latest performance metrics, or None if no data available
    """
    try:
        from data_collection.database import connect_to_db
        from psycopg2 import Error
    except ImportError:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Database module not available for fetching latest performance")
        except ImportError:
            print("Error: Database module not available for fetching latest performance")
        return None
    
    conn = None
    try:
        conn = connect_to_db()
        if conn is None:
            return None
        
        cursor = conn.cursor()
        
        # Query most recent record
        query = """
            SELECT 
                id, model_version, total_predictions, anomalies_detected,
                alerts_triggered, avg_anomaly_score, detection_rate,
                alert_rate, false_positive_rate, evaluation_period_start,
                evaluation_period_end, timestamp
            FROM model_performance
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        
        if result is None:
            return None
        
        return {
            'id': result[0],
            'model_version': result[1],
            'total_predictions': result[2],
            'anomalies_detected': result[3],
            'alerts_triggered': result[4],
            'avg_anomaly_score': float(result[5]) if result[5] is not None else None,
            'detection_rate': float(result[6]) if result[6] is not None else 0.0,
            'alert_rate': float(result[7]) if result[7] is not None else None,
            'false_positive_rate': float(result[8]) if result[8] is not None else None,
            'evaluation_period_start': result[9],
            'evaluation_period_end': result[10],
            'timestamp': result[11]
        }
    
    except Error as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Database error fetching latest performance: {e}")
        except ImportError:
            print(f"Error fetching latest performance: {e}")
        return None
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error fetching latest performance metrics: {e}", exc_info=True)
        except ImportError:
            print(f"Error fetching latest performance metrics: {e}")
        return None
    finally:
        if conn:
            conn.close()


def evaluate_model_performance(
    performance_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance against thresholds to determine if retraining is needed.
    
    This function compares current performance metrics against configurable thresholds
    to determine if the model needs retraining.
    
    Args:
        performance_data: Optional performance data dictionary. If None, fetches latest data.
    
    Returns:
        Dictionary containing:
        {
            'needs_retraining': bool,
            'reasons': List[str],  # Reasons why retraining is needed
            'current_metrics': Dict,  # Current performance metrics
            'thresholds': Dict  # Thresholds used for evaluation
        }
    """
    settings = get_settings()
    if settings is None:
        return {
            'needs_retraining': False,
            'reasons': ['Settings not available'],
            'current_metrics': {},
            'thresholds': {}
        }
    
    # Get thresholds from settings
    thresholds = {
        'min_detection_rate': settings.ml_min_detection_rate,
        'max_detection_rate': settings.ml_max_detection_rate,
        'min_alert_rate': settings.ml_min_alert_rate
    }
    
    # Fetch performance data if not provided
    if performance_data is None:
        performance_data = fetch_model_performance_data()
        if performance_data['total_records'] == 0:
            return {
                'needs_retraining': False,
                'reasons': ['No performance data available'],
                'current_metrics': {},
                'thresholds': thresholds
            }
        # Use aggregate metrics
        current_metrics = performance_data.get('aggregate_metrics', {})
    else:
        # Use provided metrics (single record or aggregate)
        if 'aggregate_metrics' in performance_data:
            current_metrics = performance_data['aggregate_metrics']
        else:
            # Single record format
            current_metrics = performance_data
    
    needs_retraining = False
    reasons = []
    
    # Check detection rate
    detection_rate = current_metrics.get('avg_detection_rate', current_metrics.get('detection_rate', 0.0))
    
    if detection_rate < thresholds['min_detection_rate']:
        needs_retraining = True
        reasons.append(
            f"Detection rate ({detection_rate:.3f}) below minimum threshold "
            f"({thresholds['min_detection_rate']:.3f}) - model too conservative"
        )
    elif detection_rate > thresholds['max_detection_rate']:
        needs_retraining = True
        reasons.append(
            f"Detection rate ({detection_rate:.3f}) above maximum threshold "
            f"({thresholds['max_detection_rate']:.3f}) - model too sensitive"
        )
    
    # Check alert rate (only if anomalies were detected)
    total_anomalies = current_metrics.get('total_anomalies_detected', current_metrics.get('anomalies_detected', 0))
    if total_anomalies > 0:
        alert_rate = current_metrics.get('avg_alert_rate', current_metrics.get('alert_rate', 0.0))
        if alert_rate is not None and alert_rate < thresholds['min_alert_rate']:
            needs_retraining = True
            reasons.append(
                f"Alert rate ({alert_rate:.3f}) below minimum threshold "
                f"({thresholds['min_alert_rate']:.3f}) - too many false positives filtered"
            )
    
    return {
        'needs_retraining': needs_retraining,
        'reasons': reasons,
        'current_metrics': current_metrics,
        'thresholds': thresholds
    }


def retrain_model(
    resource_name: Optional[str] = None,
    backup_old_model: bool = True
) -> Optional[IsolationForestDetector]:
    """
    Retrain the ML model using current historical data.
    
    This function wraps train_model_from_database() with retraining-specific
    logging and optional model backup.
    
    Args:
        resource_name: Optional resource name to train model for specific resource
        backup_old_model: If True, backup existing model before retraining
    
    Returns:
        Trained IsolationForestDetector instance if successful, None otherwise
    """
    try:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting model retraining...")
    except ImportError:
        print("Starting model retraining...")
    
    settings = get_settings()
    if settings is None:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error("Settings not available for model retraining")
        except ImportError:
            print("Error: Settings not available for model retraining")
        return None
    
    # Backup old model if requested and model exists
    if backup_old_model:
        model_path = settings.ml_model_path
        if os.path.exists(model_path):
            try:
                backup_path = f"{model_path}.backup"
                import shutil
                shutil.copy2(model_path, backup_path)
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.info(f"Backed up old model to {backup_path}")
                except ImportError:
                    print(f"Backed up old model to {backup_path}")
            except Exception as e:
                try:
                    from utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Failed to backup old model: {e}")
                except ImportError:
                    print(f"Warning: Failed to backup old model: {e}")
    
    # Retrain model
    try:
        model = train_model_from_database(resource_name=resource_name)
        
        if model is None:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.error("Model retraining failed")
            except ImportError:
                print("Error: Model retraining failed")
            return None
        
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info("Model retraining completed successfully")
        except ImportError:
            print("Model retraining completed successfully")
        
        return model
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error during model retraining: {e}", exc_info=True)
        except ImportError:
            print(f"Error during model retraining: {e}")
        return None


def check_and_retrain_model(
    resource_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check model performance and trigger retraining if needed.
    
    This function orchestrates the complete retraining workflow:
    1. Checks if retraining is enabled
    2. Fetches recent performance data
    3. Evaluates performance against thresholds
    4. Triggers retraining if performance is below thresholds
    5. Returns retraining result
    
    Args:
        resource_name: Optional resource name for resource-specific retraining
    
    Returns:
        Dictionary containing:
        {
            'retraining_triggered': bool,
            'retraining_successful': bool,
            'reasons': List[str],
            'new_model_path': Optional[str]
        }
    """
    result = {
        'retraining_triggered': False,
        'retraining_successful': False,
        'reasons': [],
        'new_model_path': None
    }
    
    settings = get_settings()
    if settings is None:
        result['reasons'].append('Settings not available')
        return result
    
    # Check if retraining is enabled
    if not settings.ml_retraining_enabled:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info("Model retraining is disabled in settings")
        except ImportError:
            print("Model retraining is disabled in settings")
        result['reasons'].append('Retraining disabled in settings')
        return result
    
    try:
        # Fetch performance data
        performance_data = fetch_model_performance_data()
        
        if performance_data['total_records'] == 0:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info("No performance data available for retraining check")
            except ImportError:
                print("No performance data available for retraining check")
            result['reasons'].append('No performance data available')
            return result
        
        # Evaluate performance
        evaluation = evaluate_model_performance(performance_data)
        
        if not evaluation['needs_retraining']:
            try:
                from utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info("Model performance is within acceptable thresholds. No retraining needed.")
            except ImportError:
                print("Model performance is within acceptable thresholds. No retraining needed.")
            result['reasons'].append('Performance within acceptable thresholds')
            return result
        
        # Performance below threshold - trigger retraining
        result['retraining_triggered'] = True
        result['reasons'] = evaluation['reasons']
        
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(
                f"Model performance below threshold. Triggering retraining. Reasons: {', '.join(evaluation['reasons'])}"
            )
        except ImportError:
            print(f"Model performance below threshold. Triggering retraining. Reasons: {', '.join(evaluation['reasons'])}")
        
        # Retrain model
        model = retrain_model(resource_name=resource_name)
        
        if model is None:
            result['reasons'].append('Retraining failed')
            return result
        
        # Retraining successful
        result['retraining_successful'] = True
        result['new_model_path'] = settings.ml_model_path
        
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info("Model retraining completed successfully")
        except ImportError:
            print("Model retraining completed successfully")
        
        return result
    
    except Exception as e:
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error in check_and_retrain_model: {e}", exc_info=True)
        except ImportError:
            print(f"Error in check_and_retrain_model: {e}")
        result['reasons'].append(f'Error: {str(e)}')
        return result


def create_ml_detector(
    detector_type: str,
    **kwargs
) -> MLAnomalyDetector:
    """
    Create an ML-based anomaly detector.
    
    Args:
        detector_type: Type of detector ('isolation_forest' or 'autoencoder')
        **kwargs: Additional arguments for detector initialization
    
    Returns:
        MLAnomalyDetector instance
    
    TODO: Implement detector factory
    """
    detectors = {
        "isolation_forest": IsolationForestDetector,
        "autoencoder": AutoencoderDetector
    }
    
    if detector_type not in detectors:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    
    return detectors[detector_type](**kwargs)
