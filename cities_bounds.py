#encoding = 'utf-8'
import os
import pytz

# This file saves the bounding box, timezone, local_directory we use to save the tweets collected in these cities

# Specify the happyplaces path
happyplaces_path = r''

# The bounding boxes of cities
# lon_min, lat_min, lon_max, lat_max
san_francisco_box = [-123.543014,36.799156,-121.536764,38.496307]
new_york_box = [-74.262714,40.492848,-73.693046,40.919761]
los_angeles_box = [-118.959390,33.699516,-117.625837,34.837462]
chicago_box = [-87.939723,41.633757,-87.520393,42.031771]
atlanta_box = [-84.559331,33.643597,-84.282973,33.890698]
boston_box = [-71.369441,42.118187,-70.849632,42.502023]
london_box = [-0.535917,51.279854,0.352602,51.682671]
netherland_box = [3.358,50.7504,7.2275,53.5552]
hong_kong_box = [113.835078,22.153388,114.406957,22.561968]
bangkok_box = [99.782410,13.383219, 101.006581,14.317204]
tokyo_box = [138.795382,35.036245,140.606039,36.339124]
singapore_box = [103.573672,1.151331,104.106509,1.480834]
riyadh_box = [46.236731,24.234349,47.368077,25.207854]
mumbai_box = [72.768807,18.886899,72.994909,19.276115]
jakarta_box = [106.685344,-6.381976,106.977405,-6.078168]
dhaka_box = [90.316881,23.659243,90.518876,23.904298]
kuala_lumpur_box = [101.192068,2.641486,101.957130,3.467673]
melbourne_box = [144.362568,-38.256869,145.788045,-37.447964]
auckland_box = [174.441513,-37.075613,175.325474,-36.651444]
wales_box = [-5.618042,51.320780,-2.645919,53.470180]
taipei_box = [121.429889,24.953611,121.674861,25.218833]
tricity_box = [18.354115,54.260080,18.948665,54.586999]
paris_box = [2.223266,48.815541,2.474577,48.907088]
vancouver_box = [-123.233099,49.195055,-123.016806,49.319192]
madrid_box = [-4.133461,40.184777,-3.212349,40.760370]
johannesburg_box = [27.935249,-26.241503,28.143009,-26.100564]
saopaulo_box = [-46.995647,-24.025362,-46.309350,-23.338671]

# The timezone of cities and local location of corresponding tweets
san_francisco_timezone, san_francisco_loc = pytz.timezone('America/Los_Angeles'), os.path.join(happyplaces_path, 'SF')
new_york_timezone, new_york_loc = pytz.timezone('America/New_York'), os.path.join(happyplaces_path, 'NY')
los_angeles_timezone, los_angeles_loc = pytz.timezone('America/Los_Angeles'), os.path.join(happyplaces_path, 'LA')
chicago_timezone, chicago_loc = pytz.timezone('America/Chicago'), os.path.join(happyplaces_path, 'Chicago')
atlanta_timezone, atlanta_loc = pytz.timezone('America/New_York'), os.path.join(happyplaces_path, 'Atlanta-Boston')
boston_timezone, boston_loc = pytz.timezone('America/New_York'), os.path.join(happyplaces_path, 'Atlanta-Boston')
london_timezone, london_loc = pytz.timezone('Europe/London'), os.path.join(happyplaces_path, 'London')
netherland_timezone, netherland_loc = pytz.timezone('Europe/Amsterdam'), os.path.join(happyplaces_path, 'Netherlands')
hong_kong_timezone, hong_kong_loc = pytz.timezone('Hongkong'), os.path.join(happyplaces_path, 'HongKong')
bangkok_timezone, bangkok_loc = pytz.timezone('Asia/Bangkok'), os.path.join(happyplaces_path, 'Bangkok')
tokyo_timezone, tokyo_loc = pytz.timezone('Asia/Tokyo'), os.path.join(happyplaces_path, 'Tokyo')
singapore_timezone, singapore_loc = pytz.timezone('Singapore'), os.path.join(happyplaces_path, 'Singapore_raw')
riyadh_timezone, riyadh_loc = pytz.timezone('Asia/Riyadh'), os.path.join(happyplaces_path, 'Riyadh-Mumbai')
mumbai_timezone, mumbai_loc = pytz.timezone('Asia/Kolkata'), os.path.join(happyplaces_path, 'Riyadh-Mumbai')
jakarta_timezone, jakarta_loc = pytz.timezone('Asia/Jakarta'), os.path.join(happyplaces_path, 'Jakata-Dhaka-Kuala')
dhaka_timezone, dhaka_loc = pytz.timezone('Asia/Dhaka'), os.path.join(happyplaces_path, 'Jakata-Dhaka-Kuala')
kuala_lumpur_timezone, kuala_lumpur_loc = pytz.timezone('Asia/Kuala_Lumpur'), os.path.join(
    happyplaces_path, 'Jakata-Dhaka-Kuala')
melbourne_timezone, melbourne_loc = pytz.timezone('Australia/Melbourne'), os.path.join(
    happyplaces_path, 'Melbourne-Auckland-Wales')
auckland_timezone, auckland_loc = pytz.timezone('Pacific/Auckland'), os.path.join(
    happyplaces_path, 'Melbourne-Auckland-Wales')
wales_timezone, wales_loc = pytz.timezone('Europe/London'), os.path.join(happyplaces_path, 'Melbourne-Auckland-Wales')
taipei_timezone, taipei_loc = pytz.timezone('Asia/Taipei'), os.path.join(happyplaces_path, 'Taipei')
tricity_timezone, tricity_loc = pytz.timezone('Europe/Warsaw'), os.path.join(happyplaces_path, 'Tricity')
paris_timezone, paris_loc = pytz.timezone('Europe/Paris'), os.path.join(happyplaces_path, 'Paris-Vancouver')
vancouver_timezone, vancouver_loc = pytz.timezone('America/Vancouver'), os.path.join(
    happyplaces_path, 'Paris-Vancouver')
madrid_timezone, madrid_loc = pytz.timezone('Europe/Madrid'), os.path.join(happyplaces_path, 'Mad-John-Sao')
johannesburg_timezone, johannesburg_loc = pytz.timezone('Africa/Johannesburg'), os.path.join(
    happyplaces_path, 'Mad-John-Sao')
saopaulo_timezone, saopaulo_loc = pytz.timezone('America/Sao_Paulo'), os.path.join(happyplaces_path, 'Mad-John-Sao')

# final cities dicts
cities_dict = {'san_francisco': [san_francisco_box, san_francisco_timezone, san_francisco_loc],
               'new_york': [new_york_box, new_york_timezone, new_york_loc],
               'los_angeles': [los_angeles_box, los_angeles_timezone, los_angeles_loc],
               'chicago': [chicago_box, chicago_timezone, chicago_loc],
               'atlanta': [atlanta_box, atlanta_timezone, atlanta_loc],
               'boston': [boston_box, boston_timezone, boston_loc],
               'london': [london_box, london_timezone, london_loc],
               'netherlands': [netherland_box, netherland_timezone, netherland_loc],
               'hong_kong': [hong_kong_box, hong_kong_timezone, hong_kong_loc],
               'bangkok': [bangkok_box, bangkok_timezone, bangkok_loc],
               'tokyo': [tokyo_box, tokyo_timezone, tokyo_loc],
               'singapore': [singapore_box, singapore_timezone, singapore_loc],
               'riyadh': [riyadh_box, riyadh_timezone, riyadh_loc],
               'mumbai': [mumbai_box, mumbai_timezone, mumbai_loc],
               'jakarta': [jakarta_box, jakarta_timezone, jakarta_loc],
               'dhaka': [dhaka_box, dhaka_timezone, dhaka_loc],
               'kuala_lumper': [kuala_lumpur_box, kuala_lumpur_timezone, kuala_lumpur_loc],
               'melbourne': [melbourne_box, melbourne_timezone, melbourne_loc],
               'auckland': [auckland_box, auckland_timezone, auckland_loc],
               'wales': [wales_box, wales_timezone, wales_loc],
               'taipei': [taipei_box, taipei_timezone, taipei_loc],
               'tricity': [tricity_box, tricity_timezone, tricity_loc],
               'paris': [paris_box, paris_timezone, paris_loc],
               'vancouver': [vancouver_box, vancouver_timezone, vancouver_loc],
               'madrid': [madrid_box, madrid_timezone, madrid_loc],
               'johannesburg': [johannesburg_box, johannesburg_timezone, johannesburg_loc],
               'sao_paulo': [saopaulo_box, saopaulo_timezone, saopaulo_loc]}






