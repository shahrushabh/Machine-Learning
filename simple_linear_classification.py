import math

""" 
    Define the list of observations that will be used as in the kNN algorithm. 
"""
observations = [
    [2,3,0,0],
    [2,0,1,0],
    [0,1,3,0],
    [0,1,2,1],
    [-1,0,1,1],
    [1,-1,1,0]
]

""" The label represents the class labels associated with each observation. """
label = ["Red","Green"]

"""
    euclidean_distance: computes the euclidean distance between two points given in form of the list below.
    @param: list1 - this list contains elements already present in data.
    @param: list2 - this list contains test elements not present in data.
    @return: returns the euclidean distance between the two points. list1 and list2
"""
def euclidean_distance(list1, list2):
    if(len(list1) >= 3 and len(list2) >= 3):
        distance = 0
        for i in range(3):
            distance += pow((list1[i]-list2[i]),2)
        return math.sqrt(distance)

"""
    predict_label: predicts the label for the new list by applying the kNN algorithm.
    @param: new_list - this list contains the test elements.
"""
def predict_label(new_list, k):
    if(len(new_list) == 3):
        # Compute all distances
        distances = []
        for index, ob in enumerate(observations):
            distances.append([euclidean_distance(ob,new_list),observations[index][3]])
        distances.sort()
        # Get the label based 
        red = green = 0
        for dist in distances[:k]:
            if(dist[1] == 0):
                red += 1
            else:
                green += 1
        # Print out the prediction.
        if(green == red):
            print "Red and Green elements are the same. So it will be randomly assigned one of the two classes."
            print "In this case it is " + str(label[distances[k][1]])
        elif(green > red):
            print "Test element is labeled as Green"
        else:
            print "Test element is labeled as Red" 

if __name__ == "__main__":
    predict_label([0,0,0],3)
