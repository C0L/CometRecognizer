#------------------------------------------------------------------------------#
# Main.py
# Cogs 108
# This is the main driver of the classification program. It only serves to parse
# command line inputs using the argparse python package which provides helpful
# functionality. It also imports our static user defined class called
# ImageParse, which provides the actual pipeline for extracting the data that is
# needed to train our regression model. 
#
# Currently there are only 3 arguments it takes.
# --split:
#   This takes in an input file and an ouput file. The input file should contain
#   a set of indivudal images, which are going to be split into their
#   components, these components will be written to the output file, along with
#   a key which will allow us to reference the olive moment of a particular
#   comet. This is mainly for constructing a data set which we can then use to
#   train whatever classification approach we are using.
#
# --categorize:
#   Placeholder. This will classify a new image with our new model
#
# --train
#   Placeholder. This will train our model with an input file
#------------------------------------------------------------------------------#
import argparse         # Parse command line arguments in orderly fashion
import ImageParse as ip # User-def class with helper methods to gather data

# This information is printed when the -h/--help flag is used
parser = argparse.ArgumentParser(description="Comet classifier")

# This screens the actual input for this set of arguments we are looking for
parser.add_argument("-s", "--split", nargs='+', dest="splitPath", 
    help="path to image file")
parser.add_argument("-c", "--categorize", dest="categorize", 
    help='find olivemoment')

# Start the actual parsing of arguments
args = parser.parse_args()

# If the splitPath variable is defined then we will proceed
if (args.splitPath):
    ip.ImageParse.serializeFile(args.splitPath[0], args.splitPath[1])
