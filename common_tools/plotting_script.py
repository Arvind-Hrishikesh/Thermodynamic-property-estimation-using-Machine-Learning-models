# This file contains all functions used to plot the data so that functions can be reused

# NEcessary imports
import matplotlib.pyplot as plt
import plotly.express as px

    
# Plotting 2D data
def plot_2D_tsne(X_data, property_name, property_values):
    cm = plt.cm.get_cmap('RdYlBu') # Colour map selected
    colour_scale_values = property_values # Colour scale values are Pc values
    pc_scatter_2D = plt.scatter(X_data[:,0], X_data[:,1], c=colour_scale_values, 
                     vmin=min(colour_scale_values), vmax=max(colour_scale_values), 
                     s=35, cmap=cm)
    plt.title(f'2D TSNE plot for {property_name}')
    plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False) # Remove axis labels
    plt.colorbar(pc_scatter_2D)
    return plt.show()


# Plotting 3D data
def plot_3D_tsne(X_data, property_name, property_values):
    fig = px.scatter_3d(x=X_data[:,0], y=X_data[:,1], z=X_data[:,2], color=property_values,
                    color_continuous_scale=px.colors.sequential.RdBu,
                    title=f'3D TSNE plot for {property_name}')
    fig.update_layout(xaxis=None)
    return fig.show()