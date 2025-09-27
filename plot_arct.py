import click
import pandas as pd
import plotly.express as px
import os

class DataVisualizer:
    input_file = None
    title = None
    df=None
    fig=None
    processed_data = None
    fit_results = None
    max_or_min=None

    """
    A class to handle data processing and visualization from CSV files.

    This class provides methods to read a CSV file, perform specific data
    cleaning operations, and generate a line plot.
    """

    def __init__(self, input_file, title=None):
        """
        Initializes the DataVisualizer with input parameters.

        Args:
            input_file (str): The path to the input CSV file.
            title (str): The title for the generated plot.
        """
        self.input_file = input_file
        self.title = title
        self.df = None
        self.processed_data = None

    def read_data(self, input_file=None):
        """
        Reads the CSV data from the specified input file.

        Returns:
            self
        """
        if input_file is None:
            input_file = self.input_file
        else:
            self.input_file = input_file
        
        click.echo(f"Reading data from {self.input_file}...")
        try:
            self.df = pd.read_csv(self.input_file)
        except Exception as e:
            click.echo(f"Error reading CSV file: {e}")

        return self
    
    def process(self,max_or_min=None):
        """
        Processes the raw DataFrame to extract and clean the data for plotting.

        This method filters for columns with a 4-digit year format, drops
        the 'Date' column, sorts by index, and converts the index to integers.

        returns self
        """
        if max_or_min is None:
            max_or_min = self.max_or_min
        else:
            self.max_or_min = max_or_min
        
        click.echo("Processing data...")
        if self.df is None:
            click.echo("No data to process!!!...")
            return self

        if max_or_min == 'max':
            dfmax = self.df.max()
        elif max_or_min == 'min':
            dfmax = self.df.min()

        # Filter for columns that have a 4-digit year format
        mask = [len(str(s)) == 4 for s in dfmax.index]
        dfmax = dfmax[mask].drop('Date', errors='ignore').sort_index()

        # Ensure the index is of integer type for proper plotting
        try:
            dfmax.index = dfmax.index.astype('int')
            self.processed_data = dfmax
        except ValueError:
            click.echo("Warning: Could not convert index to integer. Plotting may be unexpected.")
            self.processed_data = None
        
        return self
        
    def generate_plot(self, ):
        """
        Generates and saves a line plot from the processed data.

        Args:
            output_file (str): The path to save the generated image.
        """

        if self.processed_data is None or self.processed_data.empty:
            click.echo("Processed data is empty. Cannot generate plot.")
            return

        click.echo("Generating plot...")
        fig = px.scatter(
            self.processed_data, 
            title=self.title,
            trendline='ols',
        )
        results = px.get_trendline_results(fig).iloc[0,-1]
        self.fit_results = results
        b,m=results.params
        fig.add_annotation(
            text=f"y = {m}*x + {b}",
            x=0.9,y=0.9,
            ax=0,ay=0,
            xanchor='right',
            xref='paper',yref='paper',
        )
        self.fig = fig
        return self
        #self.export(output_file)
    
    def clean_fig(self):
        fig=self.fig
        fig.update_layout(
            legend=None,
            xaxis_title='Year',
            yaxis_title='Sea Ice Extent',
        )

    def export(self, output_file):
        """
        Generates and saves a line plot from the processed data.

        Args:
            output_file (str): The path to save the generated image.
        """
        # Save the figure to the specified output file
        try:
            self.fig.write_image(output_file)
            click.echo(f"Successfully saved the plot to {output_file}")
        except Exception as e:
            click.echo(f"Error saving the image: {e}")
            click.echo("Please ensure you have 'kaleido' installed for image saving: pip install kaleido")

    def forecast(self, forecast_year):
        """
        Forecasts a value and calculates the standard deviation of residuals
        for a given year using an OLS trendline.

        Args:
            forecast_year (int): The year to forecast for.
        """
        results = self.fit_results
        model = results
        
        # Predict the forecasted mean for the given year.
        # The input to predict must be a DataFrame with the same column name as the trendline x-axis.
        forecast_input = pd.DataFrame({'year': [forecast_year]})
        forecasted_mean = model.predict(forecast_input).iloc[0]
        
        # Calculate the standard deviation of the residuals.
        import numpy as np
        stdev_residuals = np.std(model.resid)

        click.echo("\n--- Forecast Results ---")
        click.echo(f"Forecasted mean for {forecast_year}: {forecasted_mean:.2f}")
        click.echo(f"Standard deviation of residuals: {stdev_residuals:.2f}")
        click.echo("------------------------\n")

def quick():
    input_file = 'arctic-sea-ice-extent.csv'
    title='youuuu'
    output_file='out.png'
    self = DataVisualizer(input_file, title).read_data().process()
    self.generate_plot()
    self.clean_fig()
    self.export(output_file)


@click.group()
def cli():
    """
    A simple CLI for generating data visualizations from CSV files.
    """
    pass

@cli.command(name='plot')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-file', '-o', type=click.Path(), default='out.png', help='The path to save the generated image.')
@click.option('--title', default=None, help='The title of the plot.')
@click.option('--max-or-min', '-m', type=str, default='max', help='max or min.')
def plot_command(input_file, output_file, title,max_or_min):
    """
    Generates a line plot from a CSV file with specific formatting.

    This command reads data from the specified INPUT_FILE, processes it to
    extract columns representing years, and then creates a line chart using
    Plotly. The resulting figure is saved to a file, with the option to
    customize the output filename and plot title.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path where the generated plot will be saved.
                           Defaults to 'out.png'.
        title (str): The title to display on the plot.
                     Defaults to filename + min/max.
    """
    # Create an instance of the DataVisualizer class and run the plot generation.
    if title is None:
        title = f"{input_file} - {max_or_min}"
    visualizer = DataVisualizer(input_file)
    visualizer.title = title
    visualizer.max_or_min = max_or_min
    visualizer.read_data()
    visualizer.process()
    visualizer.generate_plot()
    visualizer.clean_fig()
    visualizer.export(output_file)


if __name__ == '__main__':
    cli()
