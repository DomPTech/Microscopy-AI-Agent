from smolagents import Tool
import numpy as np
from gplearn.genetic import SymbolicRegressor

class SymbolicRegressionTool(Tool):
    name = "symbolic_regression"
    description = "Performs symbolic regression to find a mathematical expression that best fits the given input data features (X) to the target outputs (y)."
    inputs = {
        "x_features": {
            "type": "array",
            "description": "A 2D list or matrix of input features (X). For example: [[1.0, 2.0], [2.0, 3.0], ...]."
        },
        "y_target": {
            "type": "array",
            "description": "A 1D list or array of target values (y) corresponding to the input features. For example: [3.0, 5.0, ...]."
        },
        "feature_names": {
            "type": "array",
            "description": "Optional list of string names for the input features (e.g., ['temperature', 'pressure']).",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, x_features: list[list[float]], y_target: list[float], feature_names: list[str] = None) -> str:
        X = np.array(x_features)
        y = np.array(y_target)
        
        # Reshape X into a 2D array with one column if it's a 1D array (single feature)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Set up default feature names if not provided
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X.shape[1])]
            
        if len(feature_names) != X.shape[1]:
            raise ValueError("The number of feature_names must match the number of columns in x_features.")

        # Initialize the symbolic regressor
        est_gp = SymbolicRegressor(
            population_size=2000,
            generations=30,
            stopping_criteria=0.01,
            p_crossover=0.7, 
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05, 
            p_point_mutation=0.1,
            max_samples=0.9, verbose=0,
            parsimony_coefficient=0.001,
            random_state=42,
            feature_names=feature_names
        )
        
        try:
            est_gp.fit(X, y)
        except Exception as e:
            return f"Error during symbolic regression fitting: {str(e)}"
            
        best_program = est_gp._program
        score = est_gp.score(X, y)
        
        return (
            f"Symbolic Regression Complete.\n"
            f"Best mathematical equation found:\n{str(best_program)}\n"
            f"R^2 Score on fitting data: {score:.4f}"
        )
