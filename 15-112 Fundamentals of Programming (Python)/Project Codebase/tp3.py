# name: Anusha Bhat 
# andrewID: anushab
# 15-112 R S22




"""
Code for the game starts below!
"""
from cmu_112_graphics import *
import random
# credit note: methods used in the code below are based off of the course
# notes on the 15-112 s22 website located at:
#                                        https://www.cs.cmu.edu/~112/index.html
# cmu 112 graphics file also sourced from url above

"""
Model functions are below 
"""
 

# this function keeps tracks of all the app objects 
def appStarted(app):
    app.width = 800
    app.height = 800 
    app.paused = False 
    app.score = 0 
    app.timer = 12000
    app.timerDelay = 0
    # lines 32-35 track which page is currently open:
    app.startPage = True 
    app.instructionsPage = False
    app.menuPage = False
    app.restaurantPage = False
    app.gameOver = False  
    app.gameOverReason = None  # tracks the reason player lost the game
# menu graphics taken from: https://www.google.com/url?sa=i&url=https%3A%2F%2Fpngtree.com%2Ffreebackground%2Fcreative-kitchenware-sticker-blank-menu-poster-background_1126051.html&psig=AOvVaw1bA3JyswoddVu03Cc4dVXF&ust=1651113927067000&source=images&cd=vfe&ved=0CAwQjRxqFwoTCOCZk--cs_cCFQAAAAAdAAAAABAI
    app.menuGraphics = app.loadImage('menuGraphics.jpg')
# instructions background graphic taken from: https://www.rawpixel.com/image/4018429/illustration-psd-background-cute-frame
    app.instructionsGraphic = app.loadImage('instructionsGraphic.jpg')
# diner photo from: https://www.google.com/url?sa=i&url=https%3A%2F%2Fandreasri.com%2F&psig=AOvVaw1OFp0OFY5Cjvd9Ng0Bsi_z&ust=1651111466818000&source=images&cd=vfe&ved=0CAwQjRxqFwoTCMjk7NeTs_cCFQAAAAAdAAAAABAs
    app.dinerPic = app.loadImage('dinerPhoto.jfif')
# I made the below image
    app.restaurFloor = app.scaleImage(
                    app.loadImage('restaurFloor.jpg'), 4)
# I edited the below image from: 
        # https://imgbin.com/png/4Ys9JDyK/coffee-tables-cartoon-png
    app.tableImage = app.scaleImage(
                app.loadImage('tableImage.jpg'), 1/9)
    # app.tables is a list of locations of the various tables in the restaurant
    app.tables = [(app.width/3, 1.5 * app.height/3), (2.2 * app.width/3, 1.5 * app.height/3),
        (app.width/3, 2.5 * app.height/3), (2.2 * app.width/3, 2.5 * app.height/3)]
# the below image is from: 
# https://www.freepik.com/premium-vector/modern-kitchen-flat-color-restaurant-chef-workplace-2d-cartoon-interior-design-with-kitchenware-background-professional-cook-workspace-empty-culinary-workshop-decor_10597789.htm
    app.kitchen = app.scaleImage(
            app.loadImage('https://tinyurl.com/tpkitchen2'), 3/4)
# I edited the below image from:
    # https://ca.finance.yahoo.com/news/2009-10-01-halloween-costume-idea-flo-from-diner-dash.html
    app.waiter = app.scaleImage(
                app.loadImage('waiter.jpg'), 1/20)
    app.waiterLocation = [app.width/2, app.height/3]
# the exclaimation pic is from: https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.theguardian.com%2Fbooks%2F2009%2Fapr%2F29%2Fexclamation-mark-punctuation&psig=AOvVaw1x4HnK_bF7qufSPly-wU8R&ust=1651121087143000&source=images&cd=vfe&ved=0CAwQjRxqFwoTCMD5vse3s_cCFQAAAAAdAAAAABAD
    app.exclaimationPic = app.scaleImage(app.loadImage('exclaimation.jpg'), 1/10)
# the foodItem images are screenshots taken from:
# https://www.shutterstock.com/image-vector/different-food-dishes-vector-illustration-set-1931530862
# the below lines of code (68-83) are used to form the restaurant's menu 
    app.foodItem1 = app.scaleImage(app.loadImage('foodItem1.jpg'), 1/2)
    app.foodItem2 = app.scaleImage(app.loadImage('foodItem2.jpg'), 1/2)
    app.foodItem3 = app.scaleImage(app.loadImage('foodItem3.jpg'), 1/2)
    app.foodItem4 = app.scaleImage(app.loadImage('foodItem4.jpg'), 1/2) 
    app.foodItem5 = app.scaleImage(app.loadImage('foodItem5.jpg'), 1/2)
    app.foodItem1Resized = app.scaleImage(app.loadImage('foodItem1.jpg'), 1/6)
    app.foodItem2Resized = app.scaleImage(app.loadImage('foodItem2.jpg'), 1/6)
    app.foodItem3Resized = app.scaleImage(app.loadImage('foodItem3.jpg'), 1/6)
    app.foodItem4Resized = app.scaleImage(app.loadImage('foodItem4.jpg'), 1/6) 
    app.foodItem5Resized = app.scaleImage(app.loadImage('foodItem5.jpg'), 1/6)
    app.menuItems = [app.foodItem1, app.foodItem2, app.foodItem3, app.foodItem4,
                app.foodItem5]
    app.menuItemsResized = [app.foodItem1Resized, app.foodItem2Resized, 
                app.foodItem3Resized, app.foodItem4Resized, app.foodItem5Resized]
    app.menuItemsNames = ['Omelette', 'Hot Dogs', 'Strawberry Pancake', 
                                     "Veggie Pizza", "Kebab"]
# the customer and customerHeart images are screenshot edits from http://www.ericzimmerman.com/diner-dash
    app.customer = app.scaleImage(app.loadImage('customer.jpg'), 1/6)
    app.customerHeart = app.scaleImage(app.loadImage('customerheart.jpg'), 1/5)
# the below 3 lists are lists containing objects from the Customer class written below
    app.customersInLine = [] # customers waiting to be seated
    app.customersSeated = [] # customers who are seated but waiting to order
    app.customersOrdered = [] # customers who have ordered their food 
    app.currCustomer = None # track which customer is currently being served
    app.currFood = [] # track which food item is currently being delivered


"""
Customer Class code below 
"""


class Customer:
    def __init__(self, app): # attributes of customer object used to track customer functionalities
        customerCount = len(app.customersInLine)
        self.inGame = True 
        self.hearts = 5
        self.location = [app.width/10, 
                (17 - 3 * customerCount) * app.height/20]
        self.tableNum = None
        self.seated = False 
        self.ordered = False 
        self.served = False 
        self.orderReady = False
        self.ate = False 
        self.orderSign = False 
        self.foodNum = random.randint(0, 4)
        self.order = app.menuItemsResized[self.foodNum]
        self.foodLocation = None

# helper function used to move a customer from the line to the table they will sit at 
    def move(self, app):
        i = self.tableNum # locates the table the customer will sit at 
        tableX = app.tables[i][0] 
        tableY = app.tables[i][1]
        customerX = self.location[0]
        customerY = self.location[1]
        # conditionals on lines 126 and 130 move the customer to between the tables
        if customerY < app.height/2:
            for y in range(int(customerY), int(app.height/2) + 1, 50):
                self.location[1] = y
                customerY = y
        elif customerY >= app.height/2:
            for y in range(int(customerY), int(app.height/2 + 1), -50):
                self.location[1] = y
                customerY = y
        # move customer to the location of the table's seat's x-coordinate 
        for x in range(int(customerX), int(tableX + 1.25 * app.width/8), 50):
            self.location[0] = x
            customerX = x
        # conditionals on lines 139 and 142 finish placing the customer at the table 
        if customerY <= tableY:
            for y in range(int(customerY), int(tableY), 25):
                self.location[1] = y
        else:
            for y in range(int(customerY), int(tableY), -25):
                self.location[1] = y         
        
    def updateHearts(self, number):
        self.hearts += number 

    def updateTableNum(self, tableNum):
        self.tableNum = tableNum

    def isSeated(self):
        self.seated = True 
    
    def startOrdering(self):
        self.orderSign = True # signals that customer is ready to order

    def orderFood(self, app):
        self.orderSign = False # customer ordered so signal is off 
        self.ordered = True 
        index = self.tableNum
        self.foodLocation = [app.width/3 + index * app.width/8, app.height/7]

    # this helper function determines the location of the food on the diner counter
    # based on the table the customer is at 
    def orderMade(self):
        self.orderReady = True # food appears on counter when this is true 

    def wasServed(self):
        self.served = True
        self.orderReady = False  # once food is served, it should not appear on the counter   

    def hasEaten(self):
        self.ate = True # indicates that the customer is ready to leave the restaurant
    
    # helper function moves the customer to the restaurant's exit 
    def leaveRestaurant(self, app):
        customerX = self.location[0]
        customerY = self.location[1]
        # conditionals on lines 182 and 185 move the customer to the center of the tables 
        if customerX <= app.width/2:
            for x in range(int(customerX), int(app.width/2), 50):
                self.location[0] = x
        else:
            for x in range(int(customerX), int(app.width/2), -50):
                self.location[0] = x
        # move the customer down until they are off the canvas 
        for y in range(int(customerY), int(1.5 * app.height), 50):
            self.location[1] = y
        self.inGame = False 


"""
End of customer class
"""


# this helper function creates a master list that contains all the customer objects 
# currently in the restaurant 
def createCustomersInfoList(app):
    allCustomers = []
    for customer in app.customersInLine:
        allCustomers.append(customer)
    for customer in app.customersSeated:
        allCustomers.append(customer)
    for customer in app.customersOrdered:
        allCustomers.append(customer)
    return allCustomers
    

# this helper function resets restaurant features such as the waiter's location,
# and the player's score for example 
def resetRestaurant(app):
    app.waiterLocation = [app.width/2, app.height/3]
    app.customersInLine = []
    app.customersSeated = []
    app.customersOrdered = []
    app.score = 0
    app.gameOver = False 
    app.gameOverReason = None
    app.timerDelay = 0
    app.timer = 12000
    app.currFood = []
    app.currCustomer = None
    app.paused = False  


# this helper function updates the waiter's location to each node in the path
# from a start location to an end location determined by the path finding
# algorithm below
def moveWaiter(app, tracedPath):
    for location in tracedPath:
        app.waiterLocation = location


# this helper function is called each time the timer fires and loops through
# the customers in the restaurant to carry out their next functionality 
# (i.e., seated -> ready to order -> order is ready -> have eaten -> left)
def customerTrack(app):
    newCustomer = Customer(app) # create a new customer object 
    app.customersInLine.append(newCustomer) # new customer enters the line 
    for customer in app.customersSeated: 
        if customer.ordered == False: 
        # if customer hasn't ordered yet, they will have a sign pop up to indicate
        # they are ready to order
            customer.startOrdering()
        elif customer.ordered == True and customer.orderReady == False: 
        # if customer has ordered and the food is not ready yet, remove them from
        # the customersSeated list and place them in the customersOrdered list 
            app.customersOrdered.append(customer)
            customer.orderMade() # food appears on counter
    for customer in app.customersOrdered:
        if customer.served == True and customer.ate == False:
            # if customer was served food but has not left then they should eat
            customer.hasEaten()
        elif customer.ate == True and customer.inGame == True:
            # if customer has eaten then they should leave the restaurant
            customer.leaveRestaurant(app)
            app.score += 1 # score increase since customer was fully served
           
           
# this helper function is called every time the timer fires and removes 1
# heart from the customer's heart list (customer becomes more inpatient)
def removeHearts(app):
    customersInfo = createCustomersInfoList(app) 
    for customer in customersInfo:
        if customer.served == False: # don't take away hearts if they are eating
            customer.updateHearts(-1)
            if customer.hearts == 0: # customer leaves angry, game ends now 
                customer.leaveRestaurant(app)
                app.gameOver = True 
                app.gameOverReason = 1


# helper table that checks if a table clicked does not currently have a customer
# sitting at it 
def isEmptyTable(app, tableNum):
    customersInfo = createCustomersInfoList(app)
    for customer in customersInfo: 
        if customer.tableNum == tableNum and customer.inGame == True:
        # found a customer object that is located at this table and is still in
        # the restaurant
            return False
    return True # did not find any customers at the table 


# helper function that checks if a table clicked is empty 
def clickedOnEmptyTable(app, event):
    for tableNum in range(len(app.tables)):
        # find the table that was clicked on 
        table = app.tables[tableNum]
        tableX = table[0]
        tableY = table[1]
        # conditional checks if a table was clicked on and calls a helper function
        # to check if it is empty 
        if ((tableY - app.height/10 <= event.y <= tableY + app.height/10) and 
                (tableX - app.width/ 8 <=  event.x <= tableX + app.width/8)
                and isEmptyTable(app, tableNum)):
                return True
    return False # no empty table found where clicked 


# helper function checks if player clicked on a table 
def clickedOnTable(app, event):
    for i in range(len(app.tables)):
        # loop through the tables to determine which table, if any, was clicked
        table = app.tables[i]
        tableX = table[0]
        tableY = table[1]
        if ((tableY - app.height/10 <= event.y <= tableY + app.height/10) and 
                (tableX - app.width/ 8 <=  event.x <= tableX + app.width/8)):
                return i # clicked within bounds of a table, so return the table clicked


# this helper function checks which customer, if any, was clicked
def customerClicked(app, event):
    customersInfo = createCustomersInfoList(app)
    for customer in customersInfo:
        customerX = customer.location[0]
        customerY = customer.location[1]
        yRangeLower = event.y - 100
        yRangeUpper = event.y + 100
        xRangeLower = event.x - 100 
        xRangeUpper = event.x + 100
        # since I'm using pixels, I determined a range of pixels that the customer's  
        # image is printed on and the below conditional checks if the player
        # clicked within that range for a certain customer
        if ((yRangeLower <= customerY <= yRangeUpper) and 
                            (xRangeLower <= customerX <= xRangeUpper)):
            return customer 
    return None 


# helper function used to make a customer sit at a table 
def seatCustomer(app, event, customer):
    tableNum = clickedOnTable(app, event) # find which table was clicked on 
    customer.updateTableNum(tableNum)
    for c in range(len(app.customersInLine)):
        if app.customersInLine[c].location == customer.location:
            index = c # find index of customer clicked since you can't pop within
            # a for loop
    customer.move(app) # move customer to table 
    customer.isSeated()  # tracks that the customer has been seated 
    app.customersSeated.append(customer)
    app.customersInLine.pop(index) # allocate customer to correct list 
    if customer.hearts < 5:
        customer.updateHearts(1) # increase hearts by one since they were helped
    app.currCustomer = None # reset currCustomer 


# helper function to take a customer's order 
def takeOrder(app, customer):
    customer.orderFood(app) # customer has ordered food 
    if customer.hearts < 5:
        customer.updateHearts(1) 
    app.currCustomer = None


# helper function to determine which food order on the diner counter was clicked 
def foodClicked(app, event):
    yLower = app.height/7 - app.height/10
    yUpper = app.height/7 + app.height/10
    if (yLower <= event.y <= yUpper): # all food orders have the same y location
        for index in range(len(app.tables)):
            xLower = (app.width/3 + index * app.width/10) - app.width/10
            xUpper = (app.width/3 + index * app.width/10) + app.width/10
            if (xLower <= event.x <= xUpper): 
            # check if the event.x location is within the pixel range of a food order 
                return [event.x, event.y]
    return None # no food order clicked


# helper function that checks if the currFood is the same as the customer's order
# this prevents wrong deliveries and handles cases where diff orders are clicked but
# the food items themselves are the same so it is not a wrong delivery
def isSameOrder(app, currCustomer, foodX):
    for customer in app.customersOrdered:
        xLower = customer.foodLocation[0] - app.width/10
        xUpper = customer.foodLocation[0] + app.width/10
        if (xLower <= foodX <= xUpper):  # determines the other customer who's food was clicked
            customerFoodNum = customer.foodNum
            currCustomerFoodNum = currCustomer.foodNum
            if customerFoodNum == currCustomerFoodNum:  # compare if they have the same order
                return True 
    return False 


# helper function that checks if the food clicked for the currCustomer is the 
# correct order
def isCorrectOrder(app, customer):
    foodX = app.currFood[0]
    # if customer.orderReady == False: # currCustomer clicked's food is not ready yet
    #     if not isSameOrder(app, customer, foodX): # but if there order is the same as the food clicked then deliver anyway
    #         return False 
    xLower = customer.foodLocation[0] - app.width/10
    xUpper = customer.foodLocation[0] + app.width/10
    # all food orders have the same y location so only check the x location
    if (xLower <= foodX <= xUpper) or isSameOrder(app, customer, foodX):
        # either the current customer's food was clicked or the food is for a 
        # different customer but is the same order
        return True 
    return False 


# helper function to serve the customer their food 
def serveOrder(app, customer):
    customer.wasServed() 
    if customer.hearts < 5:
        customer.updateHearts(1)
    app.currFood = []
    app.currCustomer = None


"""
Path Finding Algorithm below
"""
# source1: referenced 15-112 and pseudocode for TA-authored TP Guide for Pathfinding located at
#           https://www.cs.cmu.edu/~112/notes/student-tp-guides/Pathfinding.pdf
# source2: referenced wikepedia page on dijsktra for pathfinding algorithm 
# and pseudocode: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
# source 3: watched graph algorithm TA-led mini leture 15-112 S22 
# source 4: watched the following video https://www.youtube.com/watch?v=pVfj6mxhdMw
# source 5: went to TA OH for conceptual understanding of algorithm
# source 6: read geeks for geeks to understand the algorithm, did not reference the 
# pseudocode: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/


# sourced the Graph class code from the TA led mini's presentation code; please 
# note that I did not write the class Graph 
class Graph:
    def __init__(self, information): 
        self.table = {} # dictionary of nodes 
        self.information = information # the only modification I made, this stores
                                       # all the information for the nodes in the map/graph
    
    def addEdge(self, nodeA, nodeB, weight): # creates an edge between two connected nodes
        if nodeA not in self.table:
            self.table[nodeA] = {}
        if nodeB not in self.table:
            self.table[nodeB] = {}
        self.table[nodeA][nodeB] = weight
        self.table[nodeB][nodeA] = weight

    def getEdge(self, nodeA, nodeB): # returns value of the edge 
        return self.table[nodeA][nodeB]

    def getNeighbors(self, node): # returns all the neighboring nodes to an input node
        return set(self.table[node])


# this helper function creates the distance dictionary and the graph of nodes needed
def createNodesDict(app):
    distance = {}
    allNodes = {
        "A": {"loc": (app.width/2, app.height/3), "neigh": {"H": 4, "B": 3, "I": 1}},
        "B": {"loc": (2.75 * app.width/3, 2 * app.height/5), "neigh": {"C": 2, "A": 3}},
        "C": {"loc": (2.75 * app.width/3, 2 * app.height/3), "neigh": {"B": 2,"D": 1}},
        "D": {"loc": (2.5 * app.width/3, 2 * app.height/3), "neigh": {"E": 1, "C": 1}},
        "E": {"loc": (app.width/2, 2 * app.height/3), "neigh": {"F": 1, "D": 1, "J": 5}},
        "F": {"loc": (app.width/3, 2 * app.height/3), "neigh": {"G": 1, "E": 1}},
        "G": {"loc": (app.width/4, 2 * app.height/3), "neigh": {"F": 1, "H": 3}},
        "H": {"loc": (app.width/4, 2 * app.height/5), "neigh": {"G": 3, "A": 4}},
        "I": {"loc": (app.width/2, 2 * app.height/5), "neigh": {"A": 1, "E": 1}},
        "J": {"loc": (app.width/2, 3.75 * app.height/4), "neigh": {"E": 5}}
            }
    allNodesGraph = Graph(allNodes) # store the nodes information as a Graph object
    for node1 in allNodes:
        location1 = allNodes[node1]["loc"]
        neighbors1 = allNodes[node1]["neigh"]
        if location1 == tuple(app.waiterLocation): # found source node, distance should = 0 
            distance[node1] = 0
        else: # note source node so distance should be infinitiy 
            distance[node1] = 100000000 # used a big number instead to symbolize infinity
        for node2 in neighbors1: 
        # loop through the neighbors of a node to add their edge to the nodes dictionary
            weight = neighbors1[node2]
            allNodesGraph.addEdge(node1, node2, weight)
    return (allNodesGraph, distance) 


# helper function that finds the target node based on the location clicked
# this ensures that the waiter is always at a node on the map
# note that I determined the target nodes based on where I would want my waiter 
# to go if a general location was clicked
def findTargetNode(app, allNodesDict, target):
    targetX = target[0]
    targetY = target[1]
    if app.height/5 < targetY < 3.5 * app.height/4 and targetX <= app.width/2:
        return "F" # between the tables on the left 
    elif app.height/5 < targetY < 3.5 * app.height/4 and targetX >= app.width/2:
        return "D" # between the tables on the right
    elif targetY >= 3.5 * app.height/4:
        return "J" # bottom of restaurant
    elif targetY <= app.height/5:
        return "A" # at the dining counter


# helper function to find the node with the shortest distance from the source node
def shortestDistance(distance, visited):
    shortestDistance = 10000000000000 # set the curr shortest distance to an arbitrarily large amount
    for node in distance:
        currDistance = distance[node]
        if currDistance < shortestDistance and node not in visited: 
        # node can't be in visited set since this function is used to find the next unvisited node
            shortestDistance = currDistance
            shortestNode = node 
    return shortestNode   


# this helper function takes in a set of parentNodes and determines the path
# that was taken to get to the target node and then returns a list of the locations
# of all the nodes in the path
def tracePath(parentNodes, targetNode, allNodes):
    tempTracedPath = [targetNode]
    finalTracedPath = []
    currNode = targetNode # start at target node and work your way back to the source node
    currParent = None
    while parentNodes != {}:
        currParent = parentNodes[currNode]
        if currParent != None:
            tempTracedPath.append(currParent)
            currNode = currParent
        else: # found source node 
            revTempTracedPath = tempTracedPath[::-1] 
            # reverse to find path from start to end rather than end to start
            for node in revTempTracedPath:
                # loop through the nodes in the path to find their locations
                location = allNodes.information[node]["loc"]
                finalTracedPath.append(location)
            return finalTracedPath


# this function implements the dijkstra algorithm to find the path with the least
# cost from a source node to a target node
def dijkstra(app, target):
    information = createNodesDict(app)  # tuple of the nodes graph and the distance dict
    allNodes = information[0]
    unvisitedNodesGraph = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"} 
    # initialize all nodes in graph as unvisited
    targetNode = findTargetNode(app, unvisitedNodesGraph, target)  
    distance = information[1]
    parentNodes = {}
    visited = set()
    while unvisitedNodesGraph != {}: # ends once we have visited all the nodes 
        currVertex = shortestDistance(distance, visited) # find the unvisited 
        # node with the shortest distance from the source node
        if distance[currVertex] == 0: # if currVertex is the source node 
            parentNodes[currVertex] = None # for the purpose of our tracePath helper function
        if currVertex == targetNode:
            # end trace once we have reached the target node
            return tracePath(parentNodes, targetNode, allNodes)
        neighbors = allNodes.getNeighbors(currVertex)
        visited = visited.union({currVertex}) # currVertex has now been visited
        unvisitedNodesGraph.remove(currVertex)
        for neighbor in neighbors:
            # loop through the neighbors and update their distance from the start
            # node accordingly
            if neighbor not in visited:
                # only check if the neighbor has not been visited
                edge = allNodes.getEdge(currVertex, neighbor)
                currDistance = distance[currVertex]
                totalDistance = edge + currDistance # distance from start node
                if totalDistance < distance[neighbor]:
                    # if distance from the start node in this path is less than
                    # the curr distance from start for the neighbor node, then
                    # updade the information below accordingly
                    distance[neighbor] = totalDistance
                    parentNodes[neighbor] = currVertex 
    

"""
Path finding algorithm end
"""


"""
Controller functions are below
"""


# this helper function will track what buttons on a keyboard the player presses
                    # to make the relevant changes to the program code 
def keyPressed(app, event):
    if  app.restaurantPage == True:
        if event.key == 'p' and app.gameOver == False:
            app.paused = not app.paused # pause game
        if event.key == 'r' and app.gameOver == True: 
            resetRestaurant(app) # reset the game 
        if (event.key == 'x' and app.startPage == False):
        # if the player exits the game mode, then the game will restart from the
        # begining, so we call resetRestaurant to do this 
            resetRestaurant(app) 
            app.startPage = True
            app.restaurantPage = False 
    elif (event.key == 'x' and (app.instructionsPage == True  or app.menuPage == True)
            and app.startPage == False):
        # return back to the start page if press 'x' when viewing instructions
        # or the menu
        app.startPage = True
        app.instructionsPage = False 
        app.menuPage = False


# this helper function will track where the player clicks the mouse on the 
                    # screen to make the relevant changes to the program code 
def mousePressed(app, event):
    # if the player clicks on the start game button on the home page, the 
    # restaurant screen will be displayed 
    if app.restaurantPage == False and app.startPage == True: 
        if (((3.5*app.width/9) <= event.x <= (4.5*app.width/9 + app.width/10)) 
            and ((app.height - app.height/4) <= event.y <= 
            (app.height - app.height/8))):
            app.restaurantPage = True
            app.startPage = False 
    # if the player clicks on the instructions button on the home page, the 
    # restaurant screen will be displayed 
    if app.instructionsPage == False and app.startPage == True:
        if (((0.5 * app.width/9) <= event.x <= (1.5*app.width/9 + app.width/10)) 
            and ((app.height - app.height/4) <= event.y <= 
            (app.height - app.height/8))):
            app.instructionsPage = True 
            app.startPage = False 
    # if the player clicks on the menu button on the home page, the 
    # restaurant screen will be displayed 
    if app.menuPage == False and app.startPage == True:
        if (((6.5 * app.width/9) <= event.x <= (7.5*app.width/9 + app.width/10)) 
            and ((app.height - app.height/4) <= event.y <= 
            (app.height - app.height/8))):
            app.menuPage = True 
            app.startPage = False 
    # if a customer is clicked, they are now the currCustomer
    if app.currCustomer == None and app.restaurantPage == True and app.paused == False:
        if customerClicked(app, event) != None:
            app.currCustomer = customerClicked(app, event)
    # if a food order is clicked, it is not the currFood item and the waiter should
    # pick up the food 
    if len(app.currFood) == 0 and app.restaurantPage == True and app.paused == False:
        if foodClicked(app, event) != None:
            foodLocation = foodClicked(app, event)
            tracedPath = dijkstra(app, foodLocation) # find path to the food
            if tracedPath != None: # only move if there is a path
                moveWaiter(app, tracedPath)
            app.currFood = foodLocation
    if app.currCustomer != None:
        # if there is a currCustomer, then the waiter must serve them
        currCustomer = app.currCustomer
        if currCustomer.seated == False and clickedOnEmptyTable(app, event):
            # if customer is not seated and an empty table was clicked, then the
            # customer should be seated
            clickedLocation = [event.x, event.y]
            tracedPath = dijkstra(app, clickedLocation) # find path to the table
            if tracedPath != None:
                moveWaiter(app, tracedPath) # move waiter along path if there is one 
            seatCustomer(app, event, currCustomer) # seat the customer
        elif currCustomer.seated == True: # customer is already seated
            customerLocation = currCustomer.location
            if currCustomer.ordered == False and currCustomer.orderSign == True:
                # move to the customer to take their order 
                tracedPath = dijkstra(app, customerLocation)
                if tracedPath != None: 
                    moveWaiter(app, tracedPath)
                takeOrder(app, currCustomer)
            elif len(app.currFood) != 0:
                if isCorrectOrder(app, currCustomer) and currCustomer.ordered == True:
                # if the customer has ordered and the currFood is the customer's order
                # then deliver the food 
                    tracedPath = dijkstra(app, customerLocation)
                    if tracedPath != None:
                        moveWaiter(app, tracedPath)
                    serveOrder(app, currCustomer)
                else:
                    # the food delivered was the wrong order so the game will end
                    app.gameOver = True 
                    app.gameOverReason = 3
                    app.currFood = []
                    

# this function runs every time the timer fires, it is mainly used to 
# periodically call the functions that can carry out the game functionalities  
def timerFired(app):
    if app.restaurantPage == False: 
        return # do not load anything if the restaurant page has not opened
    if app.paused == True:
        return # do not load anything if the game is paused
    if app.gameOver == True: 
        return # do not load anything if the game has ended
    # lines  685-690 decrease the timer according to the score such that the customers
    # enter the restaurant faster as the score increases, allowing for increasing
    # game difficulty
    if app.score <= 5:
        app.timerDelay = app.timer
    elif 5 < app.score <= 15: 
        app.timerDelay = app.timer // 2
    elif 15 < app.score <= 25:
        app.timerDelay = app.timer // 4
    elif 25 < app.score:
        app.timerDelay = app.timer // 10
    removeHearts(app) # remove one heart from each customer
    customerTrack(app) # calls the customer related functions
    if len(app.customersInLine) >= 6: # end game if too many customers are waiting in line
        app.gameOver = True
        app.gameOverReason = 2


"""
Drawing functions are below
"""


# this helper function draws the customers and their orders
def drawCustomers(app, canvas):
    customersInfo = createCustomersInfoList(app)
    for customer in customersInfo:
        customerX = customer.location[0]
        customerY = customer.location[1]
        canvas.create_image(customerX, customerY, image = 
                            ImageTk.PhotoImage(app.customer)) # customer is drawn
        numOfHearts = customer.hearts 
        for i in range(numOfHearts): # draw customer's hearts above their head
            canvas.create_image(customerX - (15 * i), customerY - 60, 
                    image = ImageTk.PhotoImage(app.customerHeart))
        if customer.orderSign == True:
            # if they are ready to order, an exlaimation mark will appear next
            # to their head
            canvas.create_oval(customerX + 15, customerY - 25, customerX + 45,
                                        customerY - 75, fill = 'white')
            canvas.create_image(customerX + 30, customerY - 50, image = 
                                        ImageTk.PhotoImage(app.exclaimationPic))            
        elif customer.ordered == True:
            food = customer.order
            if customer.served == False:
                # if the customer has ordered but they haven't been served yet,
                # then draw their order next to their head 
                cx = customerX + 15  
                canvas.create_oval(cx + 60, customerY - 25, 
                                cx  - 5, customerY - 75,
                                fill = "white", width = 0)
                canvas.create_image(cx + 30, customerY - 50, image = 
                                            ImageTk.PhotoImage(food))
                if customer.orderReady == True:
                    # if the customer's order is ready then draw the food on the
                    # diner counter
                    foodX = customer.foodLocation[0]
                    foodY = customer.foodLocation[1]
                    canvas.create_image(foodX, foodY, image =  
                                                    ImageTk.PhotoImage(food))
            elif customer.served == True and customer.ate == False:
                # if the customer was served but have not ate yet, then draw
                # their food on the table they are sitting at
                canvas.create_image(customerX - app.width/8, customerY - 15, 
                                            image =  ImageTk.PhotoImage(food))
            

# this helper function draws the restaurant page 
def drawRestaurantPage(app, canvas):
    canvas.create_image(0, 0, image = ImageTk.PhotoImage(app.restaurFloor)) # kitchen background
    canvas.create_image(app.width/2, app.height/10,
                             image = ImageTk.PhotoImage(app.kitchen)) # diner's kitchen
    for location in app.tables: # draw each table 
        x = location[0]
        y = location [1]
        canvas.create_image(x, y, 
                    image = ImageTk.PhotoImage(app.tableImage)) 
    canvas.create_rectangle(0, app.height/4, app.width/7, 
                            app.height, fill = 'light blue', width = '0')
    # ^ draw the blue carpet that the customers will stand next to
                                                # while waiting to be seated
    canvas.create_rectangle(0.82 * app.width/4,  app.height/8, 3.18 * app.width/4,
                         1.5 * app.height/7, fill = 'purple', width = '0.25')
    # ^ draw the diner counter, food orders will be placed on here when they are ready 
    waiterX = app.waiterLocation[0]
    waiterY = app.waiterLocation[1] 
    canvas.create_image(waiterX, waiterY, image = 
                                        ImageTk.PhotoImage(app.waiter)) # draw the waiter
    drawCustomers(app, canvas) # call helper function to draw the customers
    # the below line instructs the player how to go back to the home page 
    canvas.create_text(0.05 * app.width/7, app.height/50, text =
             "Press 'x' to go back", font = 'Arial 8',
                    anchor = "w") # note for player that they can press 'x' to return to the start page
    canvas.create_text(0.05 * app.width/7, 1.7 * app.height/50, text =
             "to the home page", font = 'Arial 8',
                    anchor = "w")
    canvas.create_text(0.051 * app.width/7, 2.8 * app.height/50, text =
             "Press 'p' to pause the game", font = 'Arial 8',
                    anchor = "w") # note for the player that they can press 'p' to pause the game
    canvas.create_text(6.5 * app.width/7, app.height/50, text = 
                                f'score: {app.score}', font = 'Arial 16')


# this helper function will draw the instructions by step on the instructions
# page for the player to understand the logistics of the game 
def drawInstructions(app, canvas):
    canvas.create_text(app.width/8, app.height/5, text = 
            "1. Press the start game button to begin!", fill = 'purple', 
            font = "Arial 14 bold", anchor = 'w')
    canvas.create_text(app.width/8, 2.7 * app.height/10, text = 
            "2. When a customer arrives, click on a table to seat them.",
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 3.4 * app.height/10, text = 
            "3. A red exclaimation mark will appear above a customer's head",
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 3.7 * app.height/10, text =
            "when they're ready to order. Click on them to take their order.",
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 4.4* app.height/10, text = 
            '4. Click on the order and the customer to deliver the food.',
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 5.1 * app.width/10, text = 
            "5. Customers will arrive periodically and wait in line to be served. ",
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 5.4 * app.width/10, text = 
            "As your score increases, the game will go faster.",
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 6.1 * app.height/10, text = 
            "6. Hearts above a customer's head indicate their patience level.", 
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 6.4 * app.height/10, text = 
            "Every few seconds they will lose a heart. If helped, they will gain",
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 6.7 * app.height/10, text = "a heart (max of 5).", 
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 7.4 * app.height/10, text =
            "7. The game ends once a customer loses all of their patience and",
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 7.7 * app.height/10, text = 
            'leaves, the wrong order is served, or there are too many ', 
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 8 * app.height/10, text = 
            'customers waiting in the line (max of 5 people in the line).', 
            fill = 'purple', font = "Arial 14 bold", anchor = "w")
    canvas.create_text(app.width/8, 8.7 * app.height/10, text = 
            '8. Score is based on how many customers were fully helped', 
            fill = 'purple', font = "Arial 14 bold", anchor = "w")



# this helper function draws the instructions page
def drawInstructionsPage(app, canvas):
    image = app.instructionsGraphic # background for the page
    canvas.create_image(app.width/2, app.height/2, image =
                                                ImageTk.PhotoImage(image))
    canvas.create_text(1.5 * app.width/7, app.height/50, fill = 'black',  
        text = "Press 'x' to go back to the loading screen", font = 'Arial 12')
    canvas.create_text(app.width/2, 1.5 * app.height/10, text = 'Instructions',
            font = 'Arial 30 bold', fill = 'purple')
    drawInstructions(app, canvas) # call helper function to draw instructions 


# this helper function draws the menu page
def drawMenuPage(app, canvas):
    image = app.menuGraphics # background for the page
    canvas.create_image(app.width/2, app.height/2, image =
                                                ImageTk.PhotoImage(image))
    canvas.create_text(1.5 * app.width/7, app.height/50, fill = 'white', text = 
            "Press 'x' to go back to the loading screen", font = 'Arial 12')
    # this for loop is used to draw the food images onto the menu page for the 
    # player to see what food customers can order 
    for i in range(len(app.menuItems)):
        image = app.menuItems[i]
        foodName = app.menuItemsNames[i]
        if 0 <= i % 6 <= 1: # food items 1 + 2 are in the first row  
            k = i + 1
            j = 1
        elif 2 <= i % 6 <= 3:  # food items 3 + 4 are in the second row 
            k = i - 1 
            j = 1.5
        elif 4 <= i % 6 <= 5:  # food item 5 is in the third row 
            k = i - 2.5 
            j = 2
        currX = app.width/3
        currY = 2.25 * app.height/7
        canvas.create_image(k * currX, j * currY, image = 
                                        ImageTk.PhotoImage(image)) 
        canvas.create_text(k * currX, (j + 0.25) * currY, text = foodName,
                  font = "Arial 14 bold", fill = 'purple') 


# this helper function draws the buttons on the home page
def drawButtons(app, canvas):
    # creating the buttons
    canvas.create_rectangle(0.5 * app.width/9,
            app.height - app.height/4, 1.5 * app.width/9 + app.width/10, 
            app.height - app.height/8, fill = 'white', width = 0)
    canvas.create_rectangle(3.5 * app.width/9,
            app.height - app.height/4, 4.5 * app.width/9 + app.width/10, 
            app.height - app.height/8, fill = 'white', width = 0)
    canvas.create_rectangle(6.5 * app.width/9,
            app.height - app.height/4, 7.5 * app.width/9 + app.width/10, 
            app.height - app.height/8, fill = 'white', width = 0)
     # creating the text for the buttons 
    canvas.create_text(1.5 * app.width/9, app.height - 1.5 * app.height/8, 
            text = "Instructions", font = "Arial 16 bold", 
            fill = 'purple') # gateway to the instructions page 
    canvas.create_text(4.5 * app.width/9, app.height - 1.5 * app.height/8, 
            text = "Start Game", font = "Arial 16 bold", 
            fill = 'purple') # gateway to starting the game 
    canvas.create_text(7.5 * app.width/9, app.height - 1.5 * app.height/8, 
            text = "View Menu", font = "Arial 16 bold", 
            fill = 'purple') # gateway to view the restaurant's menu


# this helper function draws the home screen of the game 
def drawStartPage(app, canvas):
    image = app.dinerPic # start page background
    canvas.create_image(app.width/2, app.height/2, image = 
                                           ImageTk.PhotoImage(image))
    # title of the game/restaurant
    canvas.create_rectangle(app.width/4, app.height/4, 3 * app.width/4, 
                        app.height/2, fill = 'white', width = 0) 
    canvas.create_text(app.width/2, app.height/3, text = "Welcome To",
            font = "Arial 26 bold", fill = 'purple')
    canvas.create_text(app.width/2, app.height/2.5, text = "Paradise Diner!",
            font = "Arial 26 bold", fill = 'purple')
    # creating the 3 buttons and the text for them 
    drawButtons(app, canvas)
   

# this helper function draws the game over message once the game has ended
def drawGameOver(app, canvas):
    canvas.create_rectangle(app.width/5, app.height/5, 4 * app.width/5, 
                        4 * app.height/5, fill = 'red', width = 0) # red box to display the message in
    canvas.create_text(app.width/2, app.height/2, text = "Game Over :(", 
                        font = "Arial 20 bold", fill = 'white') 
    canvas.create_text(app.width/2, 1.5 * app.height/2, text = 
                        'Press r to restart the game!', font = "Arial 12 bold",
                        fill = 'white') # note for player that they can press 'r' to restart the game
    canvas.create_text(app.width/2, app.height/3, text =  f'score: {app.score}',
                            font = "Arial 16 bold", fill = 'white')
    # game over reason message displayed according to the reason number
    if app.gameOverReason == 1: # customer ran out of patience/hearts
        canvas.create_text(app.width/2, 1.2 * app.height/2, text = 
                                "A customer left in anger!", 
                                font = 'Arial 12 bold', fill = 'white')
    elif app.gameOverReason == 2: # 6 customers in the line at once
        canvas.create_text(app.width/2, 1.2 * app.height/2, text = 
                                "Too many customers are waiting in the line!", 
                                font = 'Arial 12 bold', fill = 'white')
    elif app.gameOverReason == 3: # wrong order served
        canvas.create_text(app.width/2, 1.2 * app.height/2, text = 
                                "You delivered the wrong order to a customer!", 
                                font = 'Arial 12 bold', fill = 'white')

# this function calls the helper functions to draw the current graphics during each 
# point/moment of the game based on which pages are currently being displayed 
def redrawAll(app, canvas):
    if app.startPage == True: # initial state --> loading page 
        drawStartPage(app, canvas)
    # draw the restaurant page
    elif app.restaurantPage == True and app.startPage == False:
        drawRestaurantPage(app, canvas)
        if app.gameOver == True: # draws game over message
                drawGameOver(app, canvas)
    # draw the instructions page
    elif app.instructionsPage == True and app.startPage == False:
        drawInstructionsPage(app, canvas)
    # draw the menu store page 
    elif app.menuPage == True and app.startPage == False:
        drawMenuPage(app, canvas)


runApp(width=800, height=800)



