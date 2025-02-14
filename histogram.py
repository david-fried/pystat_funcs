import numpy as np
import os
import pandas as pd
import seaborn as sns


def hist_stats(data, col):

    m = round(data[col].mean(), 2)

    sd = round(data[col].std(), 2)

    n = data[col].count()

    return {'m':m,'sd':sd,'n':n}


def create_normal_dist(mean, sd, n):

    return np.random.normal(loc=mean, scale=sd, size=n)


class Histogram:
    """

    Parameters:

    -----------

    data: pandas.Dataframe

        Contains data to plot.


    """

    def __init__(self, data: pd.DataFrame) -> None:
            """

            data: pandas.Dataframe

                Contains data to plot.

            """
            self.data = data
            self.col = None
            self.group = None
       

    def attribute_error(self):

        if self.col is None:

            raise AttributeError('Histogram object has no property "col". Must set the "col" property prior to executing this method.')
    
    def _get_plot_text(self, plot_text):

        self.attribute_error()
     
        if isinstance(plot_text, str) and plot_text.lower() == 'off':

            return None
        
        if isinstance(plot_text, str):

            return plot_text

        def get_stats(df):
            m = round(df[self.col].mean(), 2)
            sd = round(df[self.col].std(), 2)
            n = df[self.col].count()
            return {'m':m,'sd':sd,'n':n}
        
        def default_text(index=None):
            df = self.data.copy()
            if index is not None:
                df = self.data.query(f"{self.group}=='{group_labels[index]}'")
            stats = get_stats(df)
            mean, sd, n = stats['m'], stats['sd'], stats['n']
            return f'M: {mean}, Sd: {sd}, N={n}'

        if self.group is not None:
            group_labels = self.data[self.group].unique()
            g1_text = group_labels[0] + '\n' + default_text(0)
            g2_text = group_labels[1] + '\n' + default_text(1)
            return g1_text + '\n\n' + g2_text
        
        return default_text()
      
    def _get_plot(self, kde):

        if self.group is not None:
            
            return sns.displot(data=self.data, x=self.col, hue=self.group, kde=kde)

        return sns.displot(self.data[self.col], kde=kde)
    
    def set_col(self, col_name: str):
        self.col = col_name
        return self
    
    def set_group(self, group_name: str):
        self.group = group_name
    
    def plot(self, **kwargs):
        """
        plot_title: str

            Title of plot. Defaults to col.

        kde: bool

            Defaults to true.

        plot_text: str

            Plot text. Defaults to Mean, SD, and N. Specify 'off' if you want to turn plot text off.

        x_pos: str

            Horizontal position of chart text.

        y_pos: str

            Vertifical position of chart text.

        fpath: str

            Path to save plot as png image. If no path exists, it will create it.
			    
        """
        self.attribute_error()
        
        plot_title = kwargs.get('plot_title', self.col)

        kde = kwargs.get('kde', True)
        
        plot_text = kwargs.get('plot_text', None)

        x_pos = kwargs.get('x_pos', None)

        y_pos = kwargs.get('y_pos', None)

        fpath = kwargs.get('fpath', None)

        plot_text = self._get_plot_text(plot_text)

        plot = self._get_plot(kde)

        plot.set(title=plot_title)

        if x_pos is not None and y_pos is not None:

            plot.ax.text(x=x_pos, y=y_pos, s=plot_text)

        else:

            y_pos = plot.ax.get_ylim()[1]

            mean = self.data[self.col].mean()

            plot.ax.text(x=mean, y=y_pos, s=plot_text)

            plot.ax.set_ylim(0, y_pos*1.1)

        if fpath is not None:

            try:

                os.makedirs(os.path.dirname(fpath), exist_ok=True)

            except Exception as e:

                print('File did not save. If saving to current working directory must specify"./" before file name')
                print(e)

            else:

                plot.savefig(fpath)