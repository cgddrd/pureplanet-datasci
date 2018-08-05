import plotly
import numpy as np
import pandas as pd
import plotly.plotly as py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from plotly import graph_objs as go

def init_plotly(username, api_key):
    plotly.tools.set_credentials_file(username=username, api_key=api_key)

def plot_stacked_funnel_chart(indata, total, colors, title, width=600):
    n_phase, n_seg = (len(indata.index), len(indata.columns))

    plot_width = width
    unit_width = plot_width / total[0]
    
    phase_w = [int(value * unit_width) for value in total]
    
    # height of a section and difference between sections 
    section_h = 100
    section_d = 10

    # shapes of the plot
    shapes = []
    
    # plot traces data
    data = []
    
    # height of the phase labels
    label_y = []

    height = section_h * n_phase + section_d * (n_phase-1)

    # rows of the DataFrame
    df_rows = list(indata.iterrows())

    # iteration over all the phases
    for i in range(n_phase):
        # phase name
        row_name = indata.index[i]
        
        # width of each segment (smaller rectangles) will be calculated
        # according to their contribution in the total users of phase
        seg_unit_width = phase_w[i] / total[i]
        seg_w = [int(df_rows[i][1][j] * seg_unit_width) for j in range(n_seg)]
        
        # starting point of segment (the rectangle shape) on the X-axis
        xl = -1 * (phase_w[i] / 2)
        
        # iteration over all the segments
        for j in range(n_seg):
            # name of the segment
            seg_name = indata.columns[j]
            
            # corner points of a segment used in the SVG path
            points = [xl, height, xl + seg_w[j], height, xl + seg_w[j], height - section_h, xl, height - section_h]
            path = 'M {0} {1} L {2} {3} L {4} {5} L {6} {7} Z'.format(*points)
            
            shape = {
                    'type': 'path',
                    'path': path,
                    'fillcolor': colors[j],
                    'line': {
                        'width': 1,
                        'color': colors[j]
                    }
            }
            shapes.append(shape)
            
            # to support hover on shapes
            hover_trace = go.Scatter(
                x=[xl + (seg_w[j] / 2)],
                y=[height - (section_h / 2)],
                mode='markers',
                marker=dict(
                    size=min(seg_w[j]/2, (section_h / 2)),
                    color='rgba(255,255,255,1)'
                ),
                text="Segment : %s" % (seg_name),
                name="Value : %d" % (indata[seg_name][row_name])
            )
            
            data.append(hover_trace)
            
            xl = xl + seg_w[j]

        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)

    label_trace = go.Scatter(
        x=[-350]*n_phase,
        y=label_y,
        mode='text',
        text=indata.index.tolist(),
        textfont=dict(
            color='rgb(200,200,200)',
            size=10
        )
    )

    data.append(label_trace)
 
    # For phase values (total)
    value_trace = go.Scatter(
        x=[350]*n_phase,
        y=label_y,
        mode='text',
        text=total,
        textfont=dict(
            color='rgb(200,200,200)',
            size=10
        )
    )

    data.append(value_trace)

    layout = go.Layout(
        title="<b>%s</b>" % title,
        titlefont=dict(
            size=20,
            color='rgb(230,230,230)'
        ),
        hovermode='closest',
        shapes=shapes,
        showlegend=False,
        paper_bgcolor='rgba(44,58,71,1)',
        plot_bgcolor='rgba(44,58,71,1)',
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            zeroline=False
        )
    )

    return go.Figure(data=data, layout=layout)


def plot_funnel_chart(data, colors, title):
    # chart stages data
    values = data.values
    phases = data.index.values

    # color of each funnel section
    colors = colors

    n_phase = len(phases)
 
    # the fixed width for the plot
    plot_width = 400
    
    # height of a section and difference between sections 
    section_h = 100
    section_d = 10
    
    # multiply factor to calculate the width of other sections
    unit_width = plot_width / max(values)
    
    # width for all the sections (phases)
    phase_w = [int(value * unit_width) for value in values]

    height = section_h * n_phase + section_d * (n_phase-1)
 
    shapes = []
    
    label_y = []
    
    for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i]/2, height, phase_w[i]/2, height - section_h]
        else:
                points = [phase_w[i]/2, height, phase_w[i+1]/2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[i],
                'line': {
                    'width': 1,
                    'color': colors[i]
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (phase name and value)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)

    # For phase names
    label_trace = go.Scatter(
        x=[-350]*n_phase,
        y=label_y,
        mode='text',
        text=phases,
        textfont=dict(
            color='rgb(200,200,200)',
            size=10
        )
    )
    
    # For phase values
    value_trace = go.Scatter(
        x=[350]*n_phase,
        y=label_y,
        mode='text',
        text=values,
        textfont=dict(
            color='rgb(200,200,200)',
            size=10
        )
    )

    data = [label_trace, value_trace]
 
    layout = go.Layout(
        title="<b>%s</b>" % title,
        titlefont=dict(
            size=20,
            color='rgb(230,230,230)'
        ),
        shapes=shapes,
        height=560,
        width=800,
        showlegend=False,
        paper_bgcolor='rgba(44,58,71,1)',
        plot_bgcolor='rgba(44,58,71,1)',
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            zeroline=False
        )
    )
    
    return go.Figure(data=data, layout=layout)

def plot_clustered_stacked(dfall, labels=None, title=None,  H="/", xfmt=None, figsize=(12, 10), **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""
    
    plt.figure(figsize=figsize)
    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=True,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify

    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    
    if not (xfmt is None):
        axe.set_xticklabels(xfmt(df.index), rotation = 0)
    else:
        axe.set_xticklabels(df.index, rotation = 0)

    if not (title is None):
        axe.set_title(title)

    # Add invisible data to add another legend
    n=[]     

    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])

    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 

    axe.add_artist(l1)

    return axe

def x_date_format(x_ticks):
    return [tick.strftime('%d\n%h\n%Y') for tick in x_ticks]

def parse_csv_date(datetext):
    return pd.datetime.strptime(datetext, '%Y-%m-%d')