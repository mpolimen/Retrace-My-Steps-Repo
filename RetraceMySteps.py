from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import random
import pickle
import numpy as np
import string
import copy


# Class for Button clickables to navigate in the program
class Button(object):
# funct is a defined function that can be called when clicked.
	def __init__(self, x, y, funct, text, color, w=100, h=50):
		self.x, self.y, self.w, self.h = x, y, w, h
		self.funct, self.text, self.color = funct, text, color

	# If the mouse clicked the button, perform its function.
	def clicked(self, event, data):
		mx, my = event.x, event.y
		x, y, w, h = self.x, self.y, self.w, self.h
		if mx > x and mx < x + w and my > y and my < y + h:
			self.funct(data)

	# Update the position to its new (x,y) coordinates.
	def changePosition(self, x, y): self.x, self.y = x, y

	# Draw the button in its color with its text.
	def draw(self, canvas):
		x, y, w, h = self.x, self.y, self.w, self.h
		canvas.create_rectangle(x, y, x + w, y + h, fill=self.color)
		canvas.create_text((2*x + w)//2, (2*y + h)//2, text=self.text)


# This class will run the drawpad and save the image for later use.
class DrawPad(object):
	def __init__(self, w, h):
		self.w, self.h = w, h

	# Animation code
	def init(self, data):
	    # load data.xyz as appropriate
	    data.addPos = [] # All colored positions
	    data.saveImage = Image.new("RGB",(data.width,data.height),(255,255,255))
	    data.pen = ImageDraw.Draw(data.saveImage)
	def mousePressed(self, event, data):
	    # use event.x and event.y
	    data.addPos.append((event.x, event.y))
	    self.savingImage(data)
	    
	def mouseDragged(self, event, data):
	    # use event.x and event.y
	    data.addPos.append((event.x, event.y))
	    self.savingImage(data)

	def redrawAll(self, canvas, data):
	    # draw in canvas
	    for i in range(len(data.addPos)):
	        x, y = data.addPos[i]
	        size = 56
	        #Will draw to the screen.
	        canvas.create_rectangle(x, y, x + size, y + size, fill = "black")
	        #Will draw to a hidden canvas that will be saved as the picture.
	        data.pen.rectangle([x, y, x+size, y + size], fill = "black")

	# Saves the image.
	def savingImage(self, data):
	    filename = "testDrawImage.png"
	    data.saveImage.save(filename)

	# Run method from course website on animation
	# http://www.cs.cmu.edu/~112n18/notes/notes-animations-part1.html#mvc
	def run(self, mainData):
		def redrawAllWrapper(canvas, data):
			canvas.delete(ALL)
			canvas.create_rectangle(0, 0, data.width, data.height,
	                                fill='white', width=0)
			self.redrawAll(canvas, data)
			canvas.update()

		def mousePressedWrapper(event, canvas, data):
			self.mousePressed(event, data)
			redrawAllWrapper(canvas, data)

		def mouseDraggedWrapper(event, canvas, data):
			self.mouseDragged(event, data)
			redrawAllWrapper(canvas, data)

		#Runs upon window exit. Will save if the user agrees to save.
		def finish():
			if messagebox.askokcancel("Save","Do you wish to use this?"):
				mainData.level.input(mainData.network.getPrediction())
				mainData.drew = mainData.network.getPrediction()
				root.destroy()

	    # Set up data and call init
		class Struct(object): pass
		data = Struct()
		data.width = self.w
		data.height = self.h
		data.timerDelay = 100 # milliseconds
		root = Tk()
		self.init(data)
	    # create the root and the canvas
		canvas = Canvas(root, width=data.width, height=data.height)
		canvas.configure(bd=0, highlightthickness=0)
		canvas.pack()
	    # set up events
		root.bind("<Button-1>", lambda event:
	                            mousePressedWrapper(event, canvas, data))
		root.bind( "<B1-Motion>", lambda event:
	                            mouseDraggedWrapper(event, canvas, data))
		# http://effbot.org/tkinterbook/tkinter-events-and-bindings.htm
		# Runs finish() when the "X" is pressed
		root.protocol("WM_DELETE_WINDOW", finish)


# Class for the level (numbers)
class Level(object):
	# Will most likely be generating a level
	def __init__(self, gen=True, moves=0, start=0, goal=0, op=None, nums=None):
		if gen:
			self.generate()
		else:
			self.moves, self.goal, self.start = moves, goal, start
			if op == None:
				self.operations, self.nums = [], []
			else:
				self.operations, self.nums = op, nums
		self.current = start
		self.index = 0
		self.currMoves = []
		self.last = []

	# Randomnly shuffles self.nums for how they are displayed to the user.
	def shuffle(self):
		# Shuffle the nums
		n = list(range(len(self.nums)))
		newPos = []
		while n != []: # Fills the newPos with a random index order
			newPos.append(n.pop(random.randint(0, len(n) - 1)))
		tempNums = []
		for spot in newPos: #Append to a random index order
			tempNums.append(self.nums[spot])
		self.nums = tempNums

	# Will randomly generate the characteristics of a level
	def generate(self, start=None, moves=None):
		# Randomly generate the other characteristics
		if start == None:
			self.moves, self.start = random.randint(3, 5), random.randint(0, 50)
		else: self.start, self.moves = start, moves
		self.goal = self.start # Start the goal at the start, change it later.
		self.operations, self.nums = [], [] # Blank them out for each call
		ops = ["+", "-", "*", "//"] # To make a random choice
		for i in range(self.moves): # For each move
			# Get a random operation and a random number.
			newOp, newNum = random.choice(ops), random.randint(0, 9)
			# Avoid division by 0 errors
			if newOp == "//" and newNum == 0: newNum = random.randint(1, 9)
			self.operations.append(newOp)
			self.nums.append(newNum)
			# Change the goal by the new operation and num
			self.goal=eval(str(self.goal)+self.operations[i]+str(self.nums[i]))
		self.shuffle() # Change the order of self.nums for the user
		self.current = self.start # The current var will be updated throughout.
		self.index, self.currMoves, self.last = 0, [], []

	# Returns the start and goal vars
	def getStartAndGoal(self): return self.start, self.goal

	def isDone(self): # Checks if the level is complete
		return self.index == self.moves and self.current == self.goal
	def isWrong(self): # Checks if the level was done wrong at the end.
		return self.index == self.moves and self.current != self.goal

	def reset(self): # Reset the level with its original characteristics
		self.current = self.start
		self.index = 0
		self.currMoves = []
		self.last = []

	def undo(self): # Undo the last move. This is why there is self.last.
		if self.last != []:
			self.current = self.last.pop()
			self.index -= 1
			self.currMoves.pop()

	# Uses recursive backtracking to find the appropriate next step for the user
	# If there is no next step, the hint will undo the last move.
	def getHint(self, total=None, i=None, steps=None):
		# Use backtracking recursion
		if total == None: # Initializes variables for recursion.
			total,i,steps=self.current,self.index,copy.copy(self.currMoves)
		# Base Case (the return value has to be anything besides None).
		if total == self.goal and i == self.moves: return total
		elif i == self.moves: return None # Will end if out of moves.
		for num in self.nums: # Go through every number to be used
			# Check if it's available and won't cause a divide by zero error.
			if (self.nums.count(num) > steps.count(num) and
			not (num == 0 and self.operations[i] == "//")):
				steps.append(num)
				# Use tmpTotal in the next call as to changing total.
				tmpTotal = eval(str(total)+self.operations[i]+str(num))
				# Recursive Call
				sol = self.getHint(tmpTotal, i + 1, steps)
				if sol != None: return num # num is the current number
				steps.remove(num)
		return None

	# Allows user to input a number to be processed in the level.
	def input(self, num):
		# First, check validity of the input.
		if num in self.nums and self.nums.count(num) > self.currMoves.count(num):
			ops = self.operations
			if num == 0 and ops[self.index] == "//":
				print("Dividing by 0 is a no-no!")
				return
			# Make the changes associated with an input.
			self.last += [self.current]
			# Updates the current value using eval().
			self.current = eval(str(self.current)+ops[self.index]+str(num))
			self.index += 1
			self.currMoves.append(num)

	# Will draw the text associated with the level.
	def draw(self, canvas, data):
		nums, op, c = str(self.nums), str(self.operations), "white"
		canvas.create_text(data.width//2,data.height//7-30,text=
		"Numbers are %s (Random order display)"%(nums),font=("Arial 18"),fill=c)
		canvas.create_text(data.width//2, data.height//6-10,text=
		"Operations are %s (Use in that order)"%(op),font=("Arial 18"),fill=c)
		canvas.create_text(data.width//2,data.height//5, 
		text="Current number: "+str(self.current), font = "Arial 18", fill=c)
		canvas.create_text(data.width//2,data.height//4, 
		text="Goal number: "+str(self.goal), font = "Arial 18", fill=c)
		canvas.create_text(data.width//2,data.height//3, 
		text="Moves done: "+str(self.currMoves), font = "Arial 18", fill=c)

# Class containing the neural network.
class Network(object):
	# Default parameters are the only parameters
	def __init__(self):
		# The trained neural network from MNIST.py is in the .sav file.
		self.net = pickle.load(open("finalized_model.sav", 'rb'))

	# Returns the image array to be used in predictions.
	# The png is changed with every savingImage call from DrawPad.
	def getImageArray(self):
		img = Image.open('testDrawImage.png', 'r')
		img = ImageOps.invert(img)
		img = img.resize((28,28), Image.ANTIALIAS)
	    # https://stackoverflow.com/questions/23935840/converting-an-rgb-image-
	    #to-grayscale-and-manipulating-the-pixel-data-in-python?rq=1
	    # Code to convert an image to grayscale and an array
		img = img.convert('L')
		imgArray=np.asarray(img.getdata(),dtype=np.float64).reshape(
	                        (img.size[1],img.size[0]))
		return imgArray


	# Referenced in MNIST.py
	# Returns a 1d array of 0's with one index with a 1 where the guess is.
	def unpack(self, data):
	    newdata = np.zeros((data.shape[0],10))
	    for i in range(len(data)):
	        newdata[i][data[i]]=1
	    return newdata

	# Flattens the single image array into a 1D array using recursion
	def flatten(self, data):
	    if isinstance(data, np.ndarray):
	        data = list(data)
	    if not isinstance(data, list):
	        return [data]
	    result = []
	    for elem in data:
	        result += self.flatten(elem)
	    return result

	# Referenced in MNIST.py
	# Uses the weights to come to a conclusion from the prediction results.
	def makeChoice(self, probs):
	    return np.argmax(probs, axis = 1)
	# Returns the prediction in a single digit for the user to understand.
	def getPrediction(self):
		imgArray = self.getImageArray()
		flatImg = np.array(self.flatten(list(imgArray))).reshape(1, -1)
		guess = self.net.predict_log_proba(flatImg)
		guess = self.unpack(self.makeChoice(guess))
		# Returns the location of the 1 index
		return(np.argwhere(guess)[0][1])

# Object to perform all handling of external files.
class FileLevels(object):
	def __init__(self):
		self.path = "CreatedLevels.txt"
		self.contents = self.readFile()

	# Basic File I/O from course website string section
	# http://www.cs.cmu.edu/~112n18/notes/notes-strings.html#basicFileIO
	def readFile(self):
	    with open(self.path, "rt") as f:
	        return f.read()
	def writeFile(self, contents):
	    with open(self.path, "wt") as f:
	        f.write(self.contents+contents)

	# Will send inputs to the level generator
	def generate(self, data):
		# in the form of "1,2*2,3*88,3*"
		pair = random.choice(self.contents[:len(self.contents) - 1].split("*"))
		start, moves = pair.split(",")[0], pair.split(",")[1]
		start, moves = int(start), int(moves)
		data.level.generate(start, moves)
################################################################################
# Main Game 
################################################################################
################################################################################
 # Button Functions
################################################################################
# Button functions for each button.
def helpB(data): data.screen = 1
def startB(data): 
	data.screen = 2
	data.mode = 0
	data.level.generate()
def homeB(data): data.screen = 0
def hintB(data):
	hint = data.level.getHint()
	if hint != None: data.level.input(hint)
	else: data.level.undo()
def drawB(data): 
	data.pad.run(data)
def nextB(data): 
	if data.mode == 0: data.level.generate()
	else: data.levelFile.generate(data)
def createB(data): 
	data.screen = 3
	data.saveString, data.typeLevel, data.currNum = "", 0, ""
def playB(data):
	data.mode, data.screen = 1, 2
	data.levelFile.generate(data)
def undoB(data): data.level.undo()
# Initializes all of the buttons
def initButtons(data):
	data.HelpButton = Button(data.width//2 - 50, data.height//4*3 - 75,helpB,
    						 "Help", "orange")
	data.StartButton = Button(data.width//2 - 50, data.height//2 - 25, startB,
    						 "Start", "green2")
	data.PlayButton = Button(data.width//2 - 50, data.height//2 - 125, playB,
    						 "Play created level", "grey")
	data.HomeButton = Button(data.width//2 - 50, data.height//4*3 - 25, homeB,
    						 "Home", "red")
	data.HintButton = Button(data.width//2 - 50, data.height//4*3 + 50, hintB,
    						 "Hint", "yellow")
	data.DrawButton = Button(data.width//2 - 50, data.height//4*3 - 100, drawB,
    						 "Draw", "blue")
	data.NextButton = Button(data.width//2 - 50, data.height//4*3 - 100, nextB,
    						 "Next", "green2")
	data.CreateButton = Button(data.width//2 - 50,data.height//4*3 + 25,createB,
    						 "Create", "brown")
	data.UndoButton = Button(data.width//2 - 250, data.height//4*3 - 25, undoB,
    						 "Undo", "purple")
################################################################################
 # Animation code
################################################################################

# Draws the rules to the screen.
def rules(canvas, data):
	fnt = "Arial 11"
	x, y = data.width//2, data.height//2
	canvas.create_text(x, y-180, text="You are given a starting number and a"+
	" goal number that you must reach. ", font=fnt)
	canvas.create_text(x,y-150,text="You are also given numbers to choose from"+
	" and which operations will be performed in order.", font=fnt)
	canvas.create_text(x,y-120,text="EX: Your start is 5 and your end is 32.",
									 font=fnt)
	canvas.create_text(x,y-90,text="Your numbers are [3, 4, 5] and your operat"+
	"ions are [+, *, +](in order).", font=fnt)
	canvas.create_text(x,y-60,text="You must add one of those numbers, then mu"+
	"ltiply, then add to get 32 from 5.", font=fnt)
	canvas.create_text(x,y-30,text="Try adding 4, multiplying by 3, then addin"+
	"g 5.", font=fnt)
	canvas.create_text(x,y,text="5+4 = 9 ---> 9*3 = 27 ---> 27+5 = 32",font=fnt)
	canvas.create_text(x,y+30,text="You can input your number selection throug"+
	"h key input or by drawing via the draw button.", font=fnt)
	canvas.create_text(x,y+60,text="When drawing it is best practice to draw c"+
	"entered and to cover most of the canvas.", font=fnt)

def init(data):
    # load data.xyz as appropriate
    data.timer = 0
    data.mode = 0 #0 is random, 1 is someone's level
    data.network = Network()
    data.screen = 0 #0 is start, 1 is help, 2 is game, 3 is create studio
    initButtons(data)
    data.level = Level()
    data.pad = DrawPad(432, 392)
    data.drew = None
    data.levelFile = FileLevels()


def mousePressed(event, data):
    if data.screen == 0: # Start Screen
    	data.HelpButton.clicked(event, data)
    	data.StartButton.clicked(event, data)
    	data.CreateButton.clicked(event, data)
    	data.PlayButton.clicked(event, data)
    elif data.screen == 1: # Help Screen
    	data.HomeButton.clicked(event, data)
    elif data.screen == 2: # Game Screen
    	data.HomeButton.clicked(event, data)
    	if data.level.isDone(): # Only if the game is done.
    		data.NextButton.clicked(event, data)
    	else: # Only during an active level.
    		data.DrawButton.clicked(event, data)
    		data.HintButton.clicked(event, data)
    		data.UndoButton.clicked(event, data)

def keyPressed(event, data):
    # use event.char and event.keysym
    if data.screen == 2: # Game Screen
    	if event.keysym in string.digits: # Must be a digit
    		data.level.input(int(event.keysym))
    		data.drew = None
    	elif event.keysym == "BackSpace": data.level.undo()
    if data.screen == 3: # Creating a level.
    	if event.keysym == "Return":
    		try: # For handling files and incorrect inputs
    			if data.typeLevel==1 and int(data.currNum)>=7:data.currNum="out"
    			data.saveString += (str(int(data.currNum))) + ","
    			data.currNum = ""
    			data.typeLevel += 1
    			if data.typeLevel == 2: #Save in form "x,y*"
    				data.screen = 0
    				data.saveString=data.saveString[:len(data.saveString)-1]+"*"
    				data.levelFile.writeFile(data.saveString)
    		except: data.screen = 0
    	elif event.keysym in string.digits: data.currNum += event.keysym
 
def timerFired(data):
    data.timer += 1
    # If the level is ever finished but wrong, reset the level.
    if data.level.isWrong():
    	data.level.reset()


# Called in redrawAll to draw all the components of the start screen.
def drawStartScreen(canvas, data):
	canvas.create_rectangle(0, 0, data.width, data.height, fill="gold")
	data.HelpButton.draw(canvas)
	data.StartButton.draw(canvas)
	data.CreateButton.draw(canvas)
	data.PlayButton.draw(canvas)
	canvas.create_text(data.width//2,data.height//9,
		text="Retrace My Steps", font="Bold 40")

def redrawAll(canvas, data):
	# Draw the buttons at the correct moments along with the appropriate text. 
    if data.screen == 0: drawStartScreen(canvas, data) # Start Screen   	
    elif data.screen == 1: # Help Screen
    	canvas.create_text(data.width//2,data.height//10,text="RULES",
    						font="Arial 30")
    	rules(canvas, data) # Draw the rules
    	data.HomeButton.draw(canvas)
    elif data.screen == 2: # Game Screen
    	canvas.create_rectangle(0,0,data.width,data.height,fill="black")
    	data.level.draw(canvas, data)
    	data.HomeButton.draw(canvas)
    	if data.level.isDone(): data.NextButton.draw(canvas)
    	else:
    		data.DrawButton.draw(canvas)
    		data.HintButton.draw(canvas)
    		data.UndoButton.draw(canvas)
    	if data.drew != None:
    		canvas.create_text(data.width//2, data.height//2, 
    							text="You drew a "+str(data.drew), fill="white")
    if data.screen == 3: 
    	canvas.create_text(data.width//2, data.height//2,text=
    	"Type a start number(enter), and how many moves under 7 (enter)!")
    	canvas.create_text(data.width//2, data.height//4*3, text=data.currNum)

# Run method from course website on animation
# http://www.cs.cmu.edu/~112n18/notes/notes-animations-part1.html#mvc
def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    root = Tk()
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

if __name__ == "__main__":
	run(600, 600)