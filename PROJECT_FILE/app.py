from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('best_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

Brand_mapping = {
    'Puma': 1, 
    'Reebok': 2, 
    'Under Armour': 3, 
    'Adidas': 4, 
    'New Balance': 5, 
    'Nike': 6
}

Category_mapping = {
    'Jeans': 1, 
    'Dress': 2, 
    'Sweater': 3, 
    'Jacket': 4, 
    'Shoes': 5, 
    'T-shirt': 6
}

Color_mapping = {
    'Black': 1, 
    'Red': 2, 
    'Green': 3, 
    'Yellow': 4, 
    'Blue': 5, 
    'White': 6
}

Size_mapping = {
    'XS': 1, 
    'S': 2, 
    'M': 3, 
    'L': 4, 
    'XL': 5, 
    'XXL': 6
}

Material_mapping = {
    'Polyester': 1, 
    'Silk': 2, 
    'Wool': 3, 
    'Denim': 4, 
    'Cotton': 5, 
    'Nylon': 6
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            brand = request.form['brand']
            category = request.form['category']
            color = request.form['color']
            size = request.form['size']
            material = request.form['material']
            
            input_data = np.array([[Brand_mapping.get(brand, -1), 
                                    Category_mapping.get(category, -1),
                                    Color_mapping.get(color, -1),
                                    Size_mapping.get(size, -1),
                                    Material_mapping.get(material, -1)]]) 

            if -1 in input_data:
                return render_template('index.html', prediction="Invalid input value. Please check your inputs.")

            prediction = model.predict(input_data)

            return render_template('index.html', prediction=prediction[0])
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
