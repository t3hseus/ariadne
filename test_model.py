import gin
from absl import flags
from absl import app

import torch

import plotly.graph_objs as go
import matplotlib.pyplot as plt

from ariadne.lightning import TrainModel
from ariadne.transformer.dataset import CloudDataset, track_params_collate_fn
from ariadne.transformer.data_loader import CloudDataLoader
from ariadne.point_net.point.points import sparse_to_points
from ariadne.transformer.spd_dataset import SPDTimesliceTracksDataset, Normalizer
# from ariadne.transformations import ConstraintsNormalize

import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='config', default=None,
    help='Path to the config file to use.'
)


#################### GETTING DATA FROM CONFIG ####################



@gin.configurable
def experiment(model,
               criterion,
               metrics,
               optimizer,
               data_loader,
               epochs,
               log_dir='lightning_logs',
               fp16_training=False,
               random_seed=None,
               accumulate_grad_batches=1,
               resume_from_checkpoint=None,  # path to checkpoint to resume
               num_gpus=None,
               clip_grads=False,
               auto_lr_find=False
               ):
    
    model = TrainModel(model=model, 
                   criterion=criterion, 
                   metrics=metrics, 
                   optimizer=optimizer, 
                   data_loader=data_loader
                   ).load_from_checkpoint(checkpoint_path='/lustre/home/user/p/photonz/ariadne/lightning_logs/PCTSegment/version_260/epoch=49-step=17600.ckpt', model=model, 
                   criterion=criterion, 
                   metrics=metrics, 
                   optimizer=optimizer, 
                   data_loader=data_loader)
    
    return model


#################### MAIN FUNCTIONS ####################

def plotly_build_plot(y_pred, y_true, hits, labels, i):
    
    # Извлекаем координаты X, Y и Z
    x_coords = hits[0, :]
    y_coords = hits[1, :]
    z_coords = hits[2, :]
    
    # Создаем трехмерную точечную диаграмму
    
    colors = ['gray' if label == -1 else 'green' for label in labels]
    names = ['Fakes' if label == -1 else 'True hits' for label in labels]

    # Create separate lists for 'Fakes' and 'True hits'
    x_fakes = [x for x, label in zip(x_coords, labels) if label == -1]
    y_fakes = [y for y, label in zip(y_coords, labels) if label == -1]
    z_fakes = [z for z, label in zip(z_coords, labels) if label == -1]

    x_true_hits = [x for x, label in zip(x_coords, labels) if label != -1]
    y_true_hits = [y for y, label in zip(y_coords, labels) if label != -1]
    z_true_hits = [z for z, label in zip(z_coords, labels) if label != -1]

    # Create the figure with two separate traces
    fig = go.Figure()

    # Add trace for 'Fakes'
    fig.add_trace(go.Scatter3d(
        x=x_fakes,
        y=y_fakes,
        z=z_fakes,
        mode='markers',
        marker=dict(
            size=5,
            color='gray',  # Customize color for 'Fakes'
            opacity=0.8
        ),
        name='Fakes'  # Legend label for 'Fakes'
    ))

    # Add trace for 'True hits'
    fig.add_trace(go.Scatter3d(
        x=x_true_hits,
        y=y_true_hits,
        z=z_true_hits,
        mode='markers',
        marker=dict(
            size=5,
            color='green',  # Customize color for 'True hits'
            opacity=0.8
        ),
        name='True hits'  # Legend label for 'True hits'
    ))

    # Update layout settings
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            title='Classes',
            orientation='v',
            x=6,  # Center the legend horizontally
            y=10,  # Center the legend vertically
            xanchor='right',  # Set the x anchor point to center
            yanchor='top'   # Set the y anchor point to middle
        )
    )
    
    fig.update_traces(textposition='top center')

    fig.update_layout(
        # height=800,
        title_text='Hits and Fakes in single event'
    )
    
    fig.add_annotation(x=0.2, y=0.05,
            text=f"PCT predict: {y_pred} TRUE: {y_true+1}",
            showarrow=False,
            # yshift=10,
            font=dict(size=18)
            )
    
    fig.update_layout(title=dict(font=dict(size=30)), legend=dict(font=dict(size=20)))
    
    fig.write_html(f"results/plot_{i}.html")
    print("Done! no.", i)


def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    
    dataset = SPDTimesliceTracksDataset(n_events_timeslice=1, n_samples=1, hits_normalizer=Normalizer())

    model = experiment()
    
    model.eval()
    
    i = 0
    for event in dataset:
        i += 1

        hits, labels, n_tracks = event
        points = sparse_to_points(X = hits, n_tracks = np.array(n_tracks))

        inputs, n_tracks = track_params_collate_fn([points])

        x = inputs['x'].to('cuda')
        mask = inputs['mask'].to('cuda')
        input_tensor = {'x': x, 'mask': mask}
        
        # y_true = torch.argmax(n_tracks, dim = 1)
        y_true = n_tracks

        output = model(input_tensor)
        y_pred = torch.argmax(output, dim = 1)
        
        print('pred:', y_pred.item(), 'fact', y_true.item())

        if i == 10: break

        plotly_build_plot(y_pred.item(), y_true.item(), hits, labels, i)
    
    

if __name__ == '__main__':
    app.run(main)