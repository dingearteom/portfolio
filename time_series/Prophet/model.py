from fbprophet import Prophet


class MyProphet(Prophet):
    def __init__(self, **kwargs):
        self.forecast = None
        super().__init__(**kwargs)

    def predict(self, df=None):
        prophet_forecast = super().predict(df)
        self.forecast = prophet_forecast
        return prophet_forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'})

    def plot_components(self):
        super().plot_components(self.forecast)
