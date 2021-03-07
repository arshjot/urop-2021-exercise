import numpy as np
from ga.chromosome_elem import ChromosomeElem
from track_generator.command import Command
from track_generator.generator import generate_track
from itertools import groupby
from config import *

np.random.seed(0)
ALLOWED_COMMANDS = [Command.S, Command.L, Command.R]


def get_random_command(max_points):
    """
    Generates a random command corresponding to S, L, and R with random length (number of points)
    and rotation (0 if command is S)

    :param max_points: The maximum number of points allowed for the command
    :return: Genotypes - List of command values and another list of corresponding rotation degrees
    """

    # Randomly select the number of points to be generated
    num_points = np.random.randint(low=1, high=max_points + 1)

    # Randomly select the value corresponding to the 3 commands (S, L, R)
    command = np.random.randint(3)
    rotation = [0]
    if (command == Command.L.value) | (command == Command.R.value):
        # If L or R is selected, we also randomly find the rotation degree
        # As we cannot have loops, max value of degree is kept as 360 / num_points
        rotation = [np.random.uniform(low=0, high=360 / num_points)]

    return [command] * num_points, rotation * num_points


def get_random_track(prev_instructions=None, prev_rotations=None, total_track_length=CHROMOSOME_LENGTH):
    """
    Generates a random track of total_track_length points.
    The generated track does not contain any loops in the middle.

    If curr_instructions and curr_rotations are provided,
    then the track is generated as a continuation of the already provided instructions.

    This function also ensures that the generated track starts and ends with an S command, however,
    this is not ensured if initial track (curr_instructions) is provided.

    :param total_track_length: The number of points in the generated track
    :param prev_instructions: The command values of the already present track which is to be completed
    :param prev_rotations: The corresponding rotation degrees to curr_instructions
    :return: Genotypes - List of command values and another list of corresponding rotation degrees.
             Length of both lists is equal to total_track_length.
    """

    if (prev_instructions is None) | (prev_rotations is None):
        # Start with S command
        curr_instructions = [Command.S.value]
        curr_rotations = [0]
    else:
        curr_instructions = prev_instructions
        curr_rotations = prev_rotations

    # If the initial part of the track has been provided, we get the current track points
    curr_track_points = generate_track(
        chromosome_elements=genotype_to_phenotype(curr_instructions, curr_rotations))
    track_length = len(curr_instructions)

    # If initial track is not provided, we generate commands for 1 point less than the required
    # because the last command has to be fixed as S
    generate_length = total_track_length - 1 if prev_instructions is None else total_track_length
    while track_length < generate_length:
        # counter to check the number of times new command resulted in a loop consecutively
        loop_counter = 0
        loop_present = True
        while loop_present & (loop_counter <= 10):

            # Get random command and update get the updated genotypes
            new_command, new_rotation = get_random_command(generate_length - track_length)
            updated_instructions = curr_instructions + new_command
            updated_rotations = curr_rotations + new_rotation

            # If initial track is not provided, we fix the last command as S
            if (prev_instructions is None) & (len(updated_instructions) == generate_length):
                updated_instructions += [Command.S.value]
                updated_rotations += [0]

            updated_track_points = generate_track(chromosome_elements=
                                                  genotype_to_phenotype(updated_instructions, updated_rotations))

            new_track_points = updated_track_points[len(curr_instructions):]
            # Check if new command results in a loop
            if len(curr_track_points) == 1:
                # cannot be a loop as only one point other than new command (which cannot have a loop)
                loop_present = False
            else:
                loop_present = creates_loop(curr_track_points, new_track_points)
            loop_counter += 1

        # If 10 random new commands result in a loop, we assume it is not possible to get a track
        # without loop from the initially selected instructions and break out to avoid infinite loop
        if loop_counter > 10:
            break

        # If new command does not result in a loop, we update our current genotypes
        curr_instructions = updated_instructions
        curr_rotations = updated_rotations
        curr_track_points = updated_track_points
        track_length = len(curr_instructions)

    # If length was less than desired (due to repeated loop creation), restart the generation process
    if (prev_instructions is None) & (len(curr_instructions) < total_track_length):
        curr_instructions, curr_rotations = get_random_track(total_track_length=total_track_length)

    return curr_instructions, curr_rotations


def genotype_to_phenotype(instructions, rotations):
    """
    Changes the genotype (lists of commands and rotations for each point)
    to phenotype (list of ChromosomeElem)

    :param instructions: List of command values
    :param rotations: List of rotations corresponding to instructions
    :return: List of ChromosomeElem objects
    """

    # Group repeated elements to get the unique command values and the corresponding number of points
    # for example - if instructions is [0, 0, 0, 1, 1, 0, 2, 2]
    # uniq_instruction_vals will contain [[0, 3], [1, 2], [0, 1], [2, 2]]
    uniq_instruction_vals = []
    for k, g in groupby(instructions):
        uniq_instruction_vals.append([k, len(list(g))])

    # Similarly, group repeated elements to get unique rotations corresponding to each command
    uniq_rotations = []
    for k, g in groupby(rotations):
        uniq_rotations.append(k)

    chromosome_elements = []
    for [inst, val], rot in zip(uniq_instruction_vals, uniq_rotations):
        # Get the actual command from command value
        command = ALLOWED_COMMANDS[inst]

        if command == Command.S:
            chromosome_elements.append(ChromosomeElem(command=command, value=val))
        else:
            # If command is L or R, we will first add a DY command with given rotation
            # and then add the L / R command
            chromosome_elements.append(ChromosomeElem(command=Command.DY, value=rot))
            chromosome_elements.append(ChromosomeElem(command=command, value=val))

    return chromosome_elements


def creates_loop(curr_points, new_points):
    """
    Checks if the new points of the track result in a loop when added to the already present track points.
    Does not check if curr_points form a loop.

    :param curr_points: List of TrackPoint objects for the initial part of the track
    :param new_points: List of TrackPoint objects for the new part (extension) of the track
    :return: Boolean indicating the presence of a loop
    """

    curr_coords = np.array([[i.x, i.y] for i in curr_points])
    new_coords = np.array([[i.x, i.y] for i in new_points])
    curr_segs = np.hstack([curr_coords[:-1], curr_coords[1:]])

    return check_loop(curr_segs, new_coords)


def has_loop(points):
    """
    Checks if the track determined by the input points contains a loop

    :param points: List of TrackPoint objects
    :return: Boolean indicating the presence of a loop
    """

    coords = np.array([[i.x, i.y] for i in points])
    curr_segs = np.concatenate([coords[0], coords[1]]).reshape(1, -1)

    return check_loop(curr_segs, coords[2:])


def check_loop(curr_segs, coords):
    """
    Used by creates_loop() and has_loop() functions to check if new segments result in a loop.

    :param curr_segs: np array with coordinates of segments already present
    :param coords: np array with coordinates of new points
    :return: Boolean indicating the presence of a loop
    """
    for idx, pt in enumerate(coords):
        # use the new point to create next segment
        new_seg = np.concatenate([curr_segs[-1][2:], pt])

        # interval covered by the new segment
        new_seg_x_int = [min(new_seg[0], new_seg[2]), max(new_seg[0], new_seg[2])]

        # get candidate segments (cand_segs) which could potentially intersect with the new segment
        # we will not be considering the previous segment as it is meant to be connected
        cand_segs = curr_segs[:-1]
        cand_segs_x_coords = np.stack([cand_segs[:, 0], cand_segs[:, 2]], axis=1)
        cand_segs_x_int = np.stack([cand_segs_x_coords.min(axis=1), cand_segs_x_coords.max(axis=1)], axis=1)

        # filter current segments with x-coord of at least one end between the x-coords of the new segment,
        # or x-coord of at least one end of the new segment between x-coords of some current segment
        # i.e., filter the segments with an overlap with the new segment in x-coords.
        # this is done to reduce the number of segments with which we need to check for intersection
        cand_segs = cand_segs[
            ((cand_segs_x_int[:, 0] >= new_seg_x_int[0]) & (cand_segs_x_int[:, 0] <= new_seg_x_int[1]) |
             (cand_segs_x_int[:, 1] >= new_seg_x_int[0]) & (cand_segs_x_int[:, 1] <= new_seg_x_int[1]) |
             (new_seg_x_int[0] >= cand_segs_x_int[:, 0]) & (new_seg_x_int[0] <= cand_segs_x_int[:, 1]) |
             (new_seg_x_int[1] >= cand_segs_x_int[:, 0]) & (new_seg_x_int[1] <= cand_segs_x_int[:, 1]))]

        # from these candidate segments, check if there is any segment
        # which intersects with the new segment
        for cand in cand_segs:
            if intersect(cand[:2], cand[2:], new_seg[:2], new_seg[2:]):
                return True

        # add the new segment to curr_segs as it does not result in a loop
        curr_segs = np.vstack([curr_segs, new_seg.reshape(1, -1)])

    return False


# Source for is_on(), collinear(), and within() functions - https://stackoverflow.com/a/328110
def is_on(a, b, c):
    """
    Return true iff point c intersects the line segment from a to b.
    (or the degenerate case that all 3 points are coincident)
    """
    return (collinear(a, b, c)
            and (within(a[0], c[0], b[0]) if a[0] != b[0] else
                 within(a[1], c[1], b[1])))


def collinear(a, b, c):
    """Return true iff a, b, and c all lie on the same line."""
    return (b[0] - a[0]) * (c[1] - a[1]) == (c[0] - a[0]) * (b[1] - a[1])


def within(p, q, r):
    """Return true iff q is between p and r (inclusive)."""
    return p <= q <= r or r <= q <= p


# ccw() and intersect() adapted from - https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def ccw(A, B, C):
    """Determines if the 3 points are listed in counterclockwise order"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """
    Returns True if line segments AB and CD intersect
    """
    # If one end of one segment lies on the other, then directly return True
    if (is_on(A, B, C)) | (is_on(A, B, D)):
        return True

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
