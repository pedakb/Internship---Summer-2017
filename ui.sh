#!/bin/bash

######################
##[global variables]##
###################### 

command_str="python main.py "
input=""

###############
##[functions]##
###############

function check_input {
	# local variables
	local is_wrong=false

	# conditional expressions	
	if $2; then # is activation
		if [[ $1 == "yes" || $1 == "no" || -z $1 ]]; then
			is_wrong=false
			
		else
			# echo "y"
			is_wrong=true
			# echo "Wrong input! Try again... "
			
		fi
	
	else  # is setting
		is_wrong=false
		
	fi	

	# return "is_wrong"
	echo $is_wrong

}

function print_error {
	if $1; then
		echo "wrong input, please try again... "
		echo	 	
	fi
}

function get_input {
	# local variables
	local is_wrong_input=true
	# echo $is_wrong_input
	# correct key loop
	while "$is_wrong_input"; do
		read -p "$1" input
		is_wrong_input=$(check_input "$input" "$2")
		# print error
		print_error $is_wrong_input
		
	done 
}

function command_update {
	# local variables
	# echo " = $2"
	# conditional expressions	
	if [ -z $2 ]; then  # is default
		# to do nothing
		echo "-> ["$1"] is set to [default]." 
		echo
		:
	
	else
		if $3; then  # is actiavation
			if [ "$2" = 'yes' ]; then
				echo "-> ["$1"] is activated."
				echo
				command_str+=' --'$1
			
			elif [ "$2" = 'no' ]; then
				echo "-> ["$1"] is not activated."
				echo
			fi

		else  # is setting
			command_str+=' --'$1' '$2
			echo "-> ["$1"] is set to ["$2"]." 
			echo
		fi
	fi

}
function print_intro {
	echo "##############################################################################################"
	echo "##############################################################################################"
	echo "###                                                                                        ###" 
	echo "### > SECURE COMMUNICATION LAB (SCL) SUMMER 2017 INTERNSHIP <                              ###"
	echo "### This is an Implementation of                                                           ###"
	echo "### (On Classification of Distorted Images with Deep Convolutional Neural Networks)        ###"
	echo "###  by (Zhou, Yiren Song, Sibo Cheung, Ngai-Man)                                          ###"
	echo "### [Which is Implemented by <Pedram Akbarian> in Summer 2017 at Secure Communication Lab] ###" 
	echo "###                                                                                        ###" 
	echo "##############################################################################################"
	echo "##############################################################################################"
	echo "Please Enter Program's Main Info. [Choose default values by pressing <ENTER>]:"
	echo

}

function print_loading {
	local spaces="                                                  "
	local bars=""
	for (( i = 1; i < 51; i++ )); do
		spaces="${spaces:1}"
		bars+="#"
		echo -ne "	Progress : [$bars$spaces  ($((2*$i)) %)]\r" 
		sleep $1
		
	done
	echo 
}

###################
##[Print Loading]##
###################
echo "	Running... (./ui.sh)"
print_loading 0.1

##################
##[Print Intro.]##
##################

print_intro

##########
##[INFO]##
##########

# Dataset
get_input '# Dataset ["MNIST" | "CIFAR-10" | "ImageNet"][default = MNIST]: ' false
command_update "data" "$input" false

# Network Arch.
get_input '# Network Arch. ["LeNet" | "CIFAR"][default = LeNet]: ' false
command_update 'arch' "$input" false

#########################
##[Learning Parameters]##
#########################

# Batch_size
get_input '# Batch Size [default = 100]: ' false
command_update 'batch_size' "$input" false

# Start_epoch
get_input '# Start Epoch [default = 0]: ' false
command_update 'start_epoch' "$input" false

# Num. of Epochs
get_input '# Num. of Epochs [default = 20]: ' false
command_update 'epochs' "$input" false

# Learning Rate
get_input '# Learning Rate [default = 0.001]: ' false
command_update 'lr' "$input" false

# Momentum
get_input '# Momentum [default = 0.9] ' false
command_update 'momentum' "$input" false

# Weight_decay
get_input '# Weight Decay [default = 0.0001]: ' false
command_update 'weight-decay' "$input" false

################
##[Distortion]##
################

get_input '# Distortion Type [motion_blur | gaussian_noise | combination | none][default = none]: ' false
command_update 'distortion' "$input" false

if [[ -z $input || $input == "none" ]]; then
	:

else
	get_input '# Distortion Level [1 : 4][default = 1]: ' false
 	command_update 'dist-level' "$input" false

fi
################
##[Processing]##
################

# Num. of Workers
get_input '# Num. Workers [default = 4]: '  false
command_update 'workers' "$input" false

# GPU
get_input '# Enter "yes" to use [GPU] processing, "no" to use [CPU] [default = "no"]: ' true
command_update 'gpu' "$input" true

##################
##[Running Mode]##
##################

# Evaluate
get_input '# Enter "yes" to evaluate, "no" to do not [default = "no"]: ' true
command_update 'evaluate' "$input" true

# Retrain
get_input '# Enter "yes" to retrain the model, "no" to do not [default = "no"]: ' true
command_update 'retrain' "$input" true

if [[ $input == 'yes' ]]; then
	# Save
	get_input '# Enter "yes" to save the model, "no" to do not [default = "no"]: ' true
	command_update 'save' "$input" true
else
	# Fine-tune
	get_input '# Enter "yes" to fine-tune, "no" to do not [default = "no"]: ' true
	command_update 'fine-tune' "$input" true
	if [[ $input == 'yes' ]]; then
		# First_N
		get_input '# First-N [default = 0]: ' false
		command_update 'first-n' "$input" false
	fi

fi

# Visualize
get_input '# Enter "yes" to visualize the prediction, "no" to do not [default = "no"]: '    
command_update 'visualize' "$input" true

# Export
get_input '# Enter "yes" to export the results, "no" to do not [default = "no"]: '    
command_update 'export' "$input" true

if [[ $input == 'yes' ]]; then
	get_input '# Enter the name of results file: ' false   
	command_update 'ex-filename' "$input" false
fi

########################
##[Command Evaluation]##
########################

echo "	Running... (./main.py)"
print_loading 0.2

eval $command_str

