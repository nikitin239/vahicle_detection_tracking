import glob
import matplotlib.image as mpimg
def getDataset():
    non_cars_files = glob.glob('/home/dnikitin/GIT/vehicledetectionandtracking/non-vehicles/non-vehicles/Extras/*.png')
    cars_files=glob.glob('/home/dnikitin/GIT/vehicledetectionandtracking/vehicles/vehicles/KITTI_extracted/*.png')
    cars = []
    not_cars = []
    for car in cars_files:
        cars.append(car)
    for non_car in non_cars_files:
        not_cars.append(non_car)
    ## Uncomment if you need to reduce the sample size
    sample_size = 6800
    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]
    print(len(cars))
    print(len(not_cars))
    ex = mpimg.imread(cars[0])
    print('Shape=',ex.shape)
    print('Type=',ex.dtype)

    # Return data_dict
    return cars, not_cars