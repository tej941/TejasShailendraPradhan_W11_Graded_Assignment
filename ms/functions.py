import pandas as pd
from ms import model
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(['M', 'B'])

def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction


def get_model_response(json_data):
    # X = pd.DataFrame.from_dict(json_data)
    # prediction = predict(X, model)
    # if prediction == 1:
    #     label = "M"
    # else:
    #     label = "B"
    # return {
    #     'status': 200,
    #     'label': label,
    #     'prediction': int(prediction)
    # }


    # Convert JSON data to DataFrame
    df = pd.DataFrame.from_dict(json_data)

    # Encode the 'diagnosis' feature if it exists
    if 'diagnosis' in df.columns:
        df['diagnosis'] = label_encoder.transform(df['diagnosis'])

    # Perform any additional preprocessing required
    # e.g., check for missing values, scale features, etc.
    
    # Ensure all columns required by the model are present
    # For example:
    expected_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                        'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                        'smoothness_se', 'compactness_se', 'concavity_se',
                        'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                        'radius_worst', 'texture_worst', 'perimeter_worst',
                        'area_worst', 'smoothness_worst', 'compactness_worst',
                        'concavity_worst', 'concave points_worst', 'symmetry_worst',
                        'fractal_dimension_worst']
    
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        return {'error': f'Missing columns: {", ".join(missing_cols)}'}, 400

    # Ensure DataFrame has the right shape
    X = df[expected_columns].fillna(0)  # Handle missing values if necessary

    # Make predictions
    prediction = predict(X, model)
    label = "M" if prediction == 1 else "B"

    return {
        'status': 200,
        'label': label,
        'prediction': int(prediction)
    }
