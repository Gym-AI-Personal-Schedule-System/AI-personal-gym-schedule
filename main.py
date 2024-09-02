from flask import Flask, request, jsonify,make_response
import pickle
import numpy as np

app = Flask(__name__)


# Load ML Model
def load_model():
    filename='model/Rf_model.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Prediction Function
def make_prediction(input_data):
    pr_val = model.predict([input_data])
    return pr_val


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from JSON
        data = request.get_json()
        age = int(data['age'])
        workout_experience = int(data['workout_experience'])
        workout_time = int(data['workout_time'])
        weight = int(data['weight'])
        height = int(data['height'])
        bmi = int(data['bmi'])
        genders = data['gender']
        fitness_goals = data['fitness_goal']

        # Numerical data
        prediction_list = [age, workout_experience, workout_time, weight, height, bmi]

        # Categorical data encoding
        gender_list = ['Female', 'Male']
        # fitness_goal_list = ['muscle up','weight loss']
        fitness_goal_list = ['General Health', 'Muscle Gain', 'Weight Gain', 'Weight Loss']

        def traverse(lst, value):
            for item in lst:
                if item == value:
                    prediction_list.append(1)
                else:
                    prediction_list.append(0)

        traverse(gender_list, genders)
        traverse(fitness_goal_list, fitness_goals)

        # Make Prediction
        pred = make_prediction(prediction_list).tolist()
        response = {'prediction': pred}
        return make_response(jsonify(response), 200)

    except Exception as e:
        response = {'error': str(e)}
        return make_response(jsonify(response), 500)

    # return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)