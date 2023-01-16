from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt

pio.renderers.default = "notebook"

def plot_point_cloud(point_cloud):
    '''
    Plot point cloud via `plotly` library.

        Args :
            point_cloud (numpy array) :  Point cloud sample with shape [n_points, x, y, z]
    '''
    data = go.Scatter3d(
        x=point_cloud[:,0],
        y=point_cloud[:,1],
        z=point_cloud[:,2],
        text=['point #{}'.format(i) for i in range(len(point_cloud))],
        mode='markers',
        marker=dict(size=1)
    )

    layout = go.Layout(
        autosize=False,
        width=500,
        height=500,
        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show(renderer='notebook')



def matplotlib_plot_point_cloud(point_cloud):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z, marker='o')
    plt.show()