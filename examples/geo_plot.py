import pandas as pd
import plotly.express as px

geofile = "~/Downloads/Sustainability_and_Transformation_Partnerships__April_2019__EN_BUC-shp/Sustainability_and_Transformation_Partnerships__April_2019__EN_BUC.shp"

data = pd.read_csv( "temp.csv", sep = ",")

t = data[ data["stp"] == "E54000007"] 
fig = px.scatter(data_frame = t, x="time", y="total_infected")
fig.write_image("fig1.png")

print(t)

