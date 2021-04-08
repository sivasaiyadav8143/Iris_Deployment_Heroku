from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model_pk = pickle.load(open('flower.pkl','rb'))

@app.route('/api_predict', methods = ['POST','GET'])
def api_predict():
    if request.method == 'GET':
        return "Please send a POST request"
    elif request.method == 'POST':
	
        print("Hello" + str(request.get_json()))
		
        data = request.get_json()
        
        sepal_length = data["sepal_length"]
        sepal_width = data["sepal_width"]
        petal_length = data["petal_length"]
        petal_width = data["petal_width"]
    
        input_data = np.array([[sepal_length, sepal_width, 
                          petal_length, petal_width]]
        

        prediction = model_pk.predict(input_data)
        return str(prediction)
    
if __name__ == "__main__":
    app.run()


        

    
   
           
       
