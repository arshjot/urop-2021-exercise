# Exercise for 2021 UROP application "Developing Self-Driving Car Autopilots using Reinforcement Learning Algorithms"

The task consists in implementing a Genetic Algorithm that generates closed tracks for the [DonkeyCar](https://www.donkeycar.com/). 
The code that generates the track starting from a list of instructions is already given (see [generator.py](https://github.com/testingautomated-usi/urop-2021-exercise/blob/master/track_generator/generator.py)).

Each [instruction](https://github.com/testingautomated-usi/urop-2021-exercise/blob/master/ga/chromosome_elem.py) is composed of a [command](https://github.com/testingautomated-usi/urop-2021-exercise/blob/master/track_generator/command.py), 
that determines the direction of the track and a value (i.e. a floating point number), that specifies the number of points for that particular command. 
The command "S" stands for straight and its associated value says the number of straight points for that track segment. The command "DY" means a rotation around
the Z axis, i.e. a curve. The value associated with "DY" indicates the amount of rotation in degrees. 
After a command "DY" there must be either a command "R" or "L" which stands respectively for right and left curve. 
The value of such commands indicates the number of points in the curve for the track arc that determine the curve.

### Example Track

Below is the track generated for the sequence of instructions \[(S 11), (DY 15.5), (R 9), (S 10)\].

<p align="center">
  <img src="https://github.com/testingautomated-usi/urop-2021-exercise/blob/master/track_example.png">
  <b>Fig.1 Example track</b><br>
</p>


The start point is indicated in green and after that there are 11 straight blue points as indicated by the instruction (S, 11). Then, there is a curve with a rotation of 15.5 degrees around the Z axis w.r.t. the straight line determined by the instruction (S, 11). The instruction indicating the rotation is (DY 15.5) which is followed by the instruction (R 9) that says to rotate for 15.5 degrees around the Z axis towards the right (with X increasing) 9 times (i.e. for a total of 139.5 degrees). After the red arc indicating the curve there is another track segment of 10 points (i.e. (S 10)) that include the end point of the track indicated in black.

## Instructions for the exercise

The exercise consists in implementing a genetic algorithm that generates a sequence of instructions for the track generator such that the resulting track forms a circuit as closed as possible. Below are the requirements that your solutions should have:

1. The fitness function needs to measure the distance between the starting point of the track (green in figure) and the end point (black). Such distance needs to be minimized;
2. The length of the sequence of instructions is fixed and indicated in the file [config.py](https://github.com/testingautomated-usi/urop-2021-exercise/blob/master/config.py) with the variable name CHROMOSOME_LENGTH;
3. The start point of the track is given and should not be changed (see the [generate.py](https://github.com/testingautomated-usi/urop-2021-exercise/blob/master/track_generator/generator.py#L15) at line 15);
4. The generated tracks should not contain any loop, i.e. the track cannot loop back onto itself. For example situations like the one in the figure below (Fig. 2) should not happen;
5. The sequence of instructions needs to start and end with an S command;
6. After a DY command there needs to be either a R command or a L command (i.e. not a S command);
7. A L or R command should be followed by either a DY or a S command

The solution to this problem is not unique, i.e. there are several tracks with the same fitness value that satisfy the requirements above. However, the fitness value of the best chromosome (i.e. sequence of instructions) produced by your genetic algorithm should be as small as possible. In any case you can visually verify your solution by plotting the points of the track to see both that start and end points are close to each other and that the track has no loops.

<p align="center">
  <img src="https://github.com/testingautomated-usi/urop-2021-exercise/blob/master/loop-example.png">
  <b>Fig.2 Example track with loop</b><br>
</p>

## Installation

Install python on your machine (we used version 3.6) and install the requirements with `pip install -r requirements.txt`.
If you execute [main.py](https://github.com/testingautomated-usi/urop-2021-exercise/blob/master/main.py) 
you should see the plot above indicating that all the dependencies have been correctly installed.
