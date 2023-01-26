from keras.models import load_model

def load_best_model():
    model_path = "./models/S1-13_G3_r_1674739698"
    model = load_model(model_path)
    return model

# if __name__ == '__main__':
#