import pickle

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', title='AI Personal Gym Schedule')


def load_model():
    filename = 'model/Rf_model.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


model = load_model()


def predict():
    try:
        data = request.get_json()
        age = int(data['age'])
        workout_experience = int(data['workout_experience'])
        workout_time = int(data['workout_time'])
        weight = int(data['weight'])
        height = int(data['height'])
        bmi = int(data['bmi'])
        gender = data['gender']

        fitness_goal = int(data['fitness_goal'])

        prediction_list = [age, workout_experience, workout_time, weight, height, bmi, gender]

        gender_list = ['Female', 'Male']
        fitness_goal_list = ['muscle up', 'weight loss']

        def traverse(lst, value):
            for item in lst:
                if item == value:
                    prediction_list.append(1)
                else:
                    prediction_list.append(0)

        traverse(gender_list, gender)
        traverse(fitness_goal_list, fitness_goal)

        pred = make_prediction(prediction_list).tolist()
        response = {'prediction': pred}

    except Exception as e:
        response = {'error': str(e)}

    print(jsonify(response))
    return jsonify(response)


def make_prediction(input_data):
    pr_val = model.predict([input_data])
    return pr_val


if __name__ == '__main__':
    app.run(debug=True)
