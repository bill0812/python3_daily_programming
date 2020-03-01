import googlemaps
from datetime import datetime

api_key = "AIzaSyDvdeHUQ6avpp2SipFq0Za6rDhRGm2Jxwg"

gmaps = googlemaps.Client(key=api_key)

elevation_result = gmaps.elevation(locations=(24.914429, 121.618209))

reverse_geocode_result_first = gmaps.reverse_geocode(latlng=(23.861485, 120.921628),location_type="ROOFTOP")

reverse_geocode_result_second = gmaps.reverse_geocode(latlng=(23.861485, 120.921628),location_type="GEOMETRIC_CENTER")

print(type(elevation_result[0]["elevation"]))

print("海拔：{} m".format(elevation_result[0]["elevation"]))

# if reverse_geocode_result_first == [] :
#     print(reverse_geocode_result_second)


print(reverse_geocode_result_second[0]["geometry"]["location"]["lat"])




