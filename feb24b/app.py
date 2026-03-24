import torch
import torch.nn as nn
import gradio as gr

model_data = torch.load('model.pth')

fm = model_data['fm']
fs = model_data['fs']
parameters = model_data['parameters']

linear = nn.Linear(3,1)             #linear part of nn, doesn't include sigmoid
linear.load_state_dict(parameters)  

model = nn.Sequential(linear, nn.Sigmoid())    #creating the model with 2 layers, a linear layer and a sigmoid layer

features = torch.tensor([
    [53.0, 242.0, 136.0]
]).float()

X = (features - fm) / fs
classification = model(X)   # prints probability


def f(hours,exams):                 #function to be used in gr.Blocks, some diffs
    features = torch.tensor([
        [hours, exams]
    ]).float()

    X = (features - fm) / fs
    classification = model(X).item()
    return "Pass" if classification >= 0.5 else "Fail"

with gr.Blocks() as iface:
    hours_box = gr.Number(label = "Provide study hours")
    exams_box = gr.Number(label = "Provide practice exams")
    result_box = gr.Textbox(label = "Result")
    hours_box.change(fn = f, inputs = [hours_box, exams_box], outputs = [result_box])
    exams_box.change(fn = f, inputs = [hours_box, exams_box], outputs = [result_box])

iface.launch()

