########
# Gradio application of 2/17 regression prediction practice from class
#########

import torch
import torch.nn as nn
import gradio as gr

model_data = torch.load('model.pth')    #loading the model we created before

fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']

parameters = model_data['parameters']      #weights and bias
model = nn.Linear(2,1)                     #2 features, 1 target
model.load_state_dict(parameters)       #loads the saved weights/bias into the model (saved from parameters)

def f(weight, engine_size):             #defining a function f to be used later in gr.Blocks
    features = torch.tensor([
        [weight, engine_size]
    ]).float()

    X = (features-fm)/fs
    Yhat = model(X)
    prediction = (Yhat*ts) + tm
    return prediction.item()        # returns just the number, not in a tensor


with gr.Blocks() as iface:
    weight_box = gr.Number(label = "Provide weight of vehicle")
    engine_box = gr.Number(label = "Provide engine size")
    mpg_box = gr.Number(label = "MPG prediction")
    weight_box.change(fn = f, inputs = [weight_box, engine_box], outputs = [mpg_box])
    engine_box.change(fn = f, inputs = [weight_box, engine_box], outputs = [mpg_box])

iface.launch()


