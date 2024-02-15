import argparse
import numpy as np
import random
import re
import scipy.stats
import sklearn.metrics
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from itertools import repeat
from itertools import product

# regular expressions
comment_re = re.compile("#.*")
numeric_re = re.compile('\d+(\.\d+)?')

# accepts a candidate solution(xyz coordinates of all loci) can have shape (3n,) or (n,3)
# returns the distance matrix between all of said loci
def sol2distVec(sol):
    # Ensuring xyz_col to have shape (n,3)
    if len(np.shape(sol)) == 1:
        xyz_col = np.array(np.split(sol, len(sol)/3))
    else:
        xyz_col = sol
    n = len(xyz_col)
    # we construct rows_m, cols_m to have shape (n,n,3)
    rows = np.reshape(xyz_col, (n, 1, 3))
    rows_m = np.tile(rows, (1, n, 1))
    cols = np.reshape(xyz_col, (1, n, 3))
    cols_m = np.tile(cols, (n, 1, 1))
    return np.sum((cols_m - rows_m)**2, 2)**0.5

# accepts contact matrix and scaling parameter alpha
# returns the distance matrix
def contactToDistance(contact, alpha=0.5):
    with np.errstate(divide="ignore"):
        matrix = 1 / (contact**alpha)
    return matrix

# local search
def local_search(initial_solution, loss_function, max_iter=1000, step_size=0.1):
    current_solution = initial_solution
    current_loss = loss_function(initial_solution)
    for _ in range(max_iter):
        # Generate a new candidate solution by perturbing the current solution
        perturbation = np.random.normal(scale=step_size, size=current_solution.shape)
        candidate_solution = current_solution + perturbation
        candidate_loss = loss_function(candidate_solution)
        # Move to the new solution if it improves the loss
        if candidate_loss < current_loss:
            current_solution = candidate_solution
            current_loss = candidate_loss
    return current_solution

# bat algorithm with local search
class Bat:
    # Parameters for bat algorithm
    params = set(["matrix", "structs", "alpha", "func", "num_bats", "upper_bound", "lower_bound",
                  "min_freq", "max_freq", "volume", "generations", "pulse", "perturbation"])

    def __init__(self, matrix, num_bats=20, lower_bound=0, upper_bound=10, min_freq=0, max_freq=2,
                 volume=0.5, generations=100, pulse=0.5, perturbation=0.01):
        self.lower = lower_bound 
        self.upper = upper_bound 
        self.freq_min = min_freq
        self.freq_max = max_freq
        self.D = len(matrix) * 3
        self.matrix = matrix
        self.generations = int(generations)  # how many iterations of the algorithm
        self.perturbation = perturbation  # this dictates the size of a bat's random walk after pulsing
        self.loudness = volume
        self.num_bats = int(num_bats)
        self.best_fit = np.Inf
        self.pulse = pulse
        self.round = lambda x: 0 if x >= self.pulse else 1
        self.freq = np.zeros(self.num_bats)
        self.sols = self.lower + (self.upper - self.lower) * np.random.rand(self.num_bats, self.D)
        self.velo = np.zeros((self.num_bats, self.D))  # bats velocity vector with shape (num_bats,D), where D is the number of dimensions 
        self.fitness = np.apply_along_axis(self.loss, 1, self.sols)
        self.evalBats()

    def loss(self, sol):
        key = self.matrix
        sum_matrix = sol2distVec(sol)
        loss_matrix = key - sum_matrix
        loss_matrix = np.nan_to_num(loss_matrix, copy=True, posinf=0)
        return np.sum(np.absolute(np.tril(loss_matrix)))

    def evalBats(self):
        best_bat = np.argmin(self.fitness)
        if self.fitness[best_bat] <= self.best_fit:
            self.best_sol = self.sols[best_bat]
            self.best_fit = self.fitness[best_bat]

    def fly(self):
        for g in range(self.generations):
            self.freq = self.freq_min + (self.freq_max - self.freq_min) * np.random.rand(self.num_bats)
            self.velo = self.velo + (self.sols - self.best_sol) * self.freq[:, np.newaxis]
            self.temp_sols = self.sols + self.velo
            self.temp_sols = np.clip(self.temp_sols, self.lower, self.upper)

            self.pulse_array = (np.array([self.pulse] * self.num_bats) >= np.random.rand(self.num_bats)).astype(int)
            self.flip_pulse_array = (self.pulse_array + 1) % 2
            self.temp_sols = (self.temp_sols * self.flip_pulse_array[:, np.newaxis]) + (
                        np.tile(self.best_sol, (self.num_bats, 1)) * self.pulse_array[:, np.newaxis] +
                        self.perturbation * np.random.normal(0, 1, (self.num_bats, self.D)))
            self.temp_sols = np.clip(self.temp_sols, self.lower, self.upper)

            self.temp_fitness = np.apply_along_axis(self.loss, 1, self.temp_sols)

            self.accepted_sols = (self.temp_fitness <= self.fitness).astype(int) * (
                        np.array([self.loudness] * self.num_bats) >= np.random.rand(self.num_bats)).astype(int)
            self.declined_sols = (self.accepted_sols + 1) % 2
            self.sols = (self.sols * self.declined_sols[:, np.newaxis]) + (
                        self.temp_sols * self.accepted_sols[:, np.newaxis])
            self.fitness = (self.fitness * self.declined_sols) + (self.temp_fitness * self.accepted_sols)
            self.evalBats()

            # Introduce local search
            for i in range(self.num_bats):
                self.sols[i] = local_search(self.sols[i], self.loss)

        return self.best_sol

# Process input function remains unchanged

# Driver code remains unchanged
def removeZeroRowCol(matrix):
	cols_nonzero=np.where(~np.all(np.isclose(matrix,0),axis=0))[0]#finds nonzero cols
	rows_nonzero=np.where(~np.all(np.isclose(matrix,0),axis=1))[0]#finds nonzero rows
	rowcol_nonzero=list(set(cols_nonzero)|set(rows_nonzero))#finds nonzero rol/col pairs
	rowcol_nonzero=np.sort(rowcol_nonzero)
	if len(rowcol_nonzero)==0:
		raise ValueError("Input Matrix is all zero...")
	new_matrix=matrix[rowcol_nonzero[:,np.newaxis],rowcol_nonzero]#removes zero row/cols
	index_dict={old:new for new,old in enumerate(rowcol_nonzero)}#build old to new index dictionary
	return (new_matrix,index_dict)


#accepts symmetric square matrix a
#returns a with zero adjancent bins replaced with an average adjancency value
def adjancenyPreprocess(a):
	a=a.copy() #method is nondestructive
	adj_diagonal=np.diagonal(a,1).copy() #this is the diagonal of the matrix off by 1
	mean_adj=sum(adj_diagonal)/len(adj_diagonal[adj_diagonal != 0]) #mean of all values on off diagonal not including 0s
	adj_diagonal[adj_diagonal == 0] = mean_adj #this is corrected off diagonal
	a.flat[1::len(a)+1]=adj_diagonal #this fixes the diagonal to the right of the main diagonal
	a.flat[len(a)::len(a)+1]=adj_diagonal #this fixes the diagonal to the left of the main diagonal
	return a

#used for making the pdb files
#pads the input string so it is size long
#left=True pads on the left False on the right
def pad(value,size,left=False):
	value=str(value)
	space_to_add=size-len(value)
	if left:
		return space_to_add*" "+value
	else:
		return value+space_to_add*" "
#writes pdb based on input solution and an outfile name
def outputPdb(input_sol,outfile=None):
	if outfile is None:
		outfile="bat.pdb"
	else:
		outfile+=".pdb"
	sol=np.array(input_sol)
	if len(np.shape(sol))==1:
		sol=np.array(np.split(sol,len(sol)/3))
	if len(np.shape(sol))!=2:
		raise ValueError(f"Invalid solution shape {np.shape(input_sol)}")

	out_string="pdb carefully constructed by Brandon\n"
	for i in range(len(sol)):
		x,y,z=sol[i]
		out_string+="ATOM  "+pad(i+1,5,True)+"   " \
			+"CA MET "+pad("B"+str(i+1),6)+"   " \
			+pad(format(x,".3f"),8,True)+pad(format(y,".3f"),8,True) \
			+pad(format(z,".3f"),8,True)+"  " \
			+"0.20 10.00\n"
	connect_string="\n".join([f"CONECT"+pad(i+1,5,True)+pad(i+2,5,True) for i in range(len(sol)-1)])
	out_string+=connect_string+"\n"+"END\n"

	f=open(outfile,"w")
	f.write(out_string)
	f.close()
#this does all the command line processing
#return (contact matrix,alpha,bat algo parameters,outfile,structs)
def processInput(input_fname,parameter_fname=None):
	contact=np.genfromtxt(input_fname)
	alpha=[]
	perturbation=[]
	structs=1
	outfile=None
	if parameter_fname is None:
		return (contact,alpha,perturbation,dict(),outfile,structs)
	pfile=open(parameter_fname,"r")
	plines=pfile.readlines()
	param_dict=dict()
	
	for line in plines:
		line=re.sub(comment_re,"",line)
		line="".join(line.split())
		line=line.replace("\t","")
		if line=="":
			continue
		param,arg=line.split("=")
		if param=="output_file":
			outfile=arg
			continue
		elif param not in Bat.params:
			raise ValueError(f"{param} is not a valid parameter")
		elif param=="structs":
			structs=int(arg)
		elif numeric_re.match(arg) is None:
			raise ValueError(f"{arg} must be numeric")
		elif param=="alpha":
			alpha=intrepretCommaSeperated(arg)
		elif param=="perturbation":
			perturbation=intrepretCommaSeperated(arg)
		else:
			param_dict[param]=float(arg)

	# distance_matrix=contactToDistance(contact,alpha)
	return (contact,alpha,perturbation,param_dict,outfile,structs)

#accepts string with comma seperated values
#returns list of floats
#None if no float 
def intrepretCommaSeperated(string:str):
	vals=string.split(",")
	vals=[v for v in vals if v] #gets rid of empty strings - note the empty string uniquely evaluates as false
	return list(map(float,vals))


#accepts the proposed solution and the target distance matrix
#returns a string with PCC,SCC,RMSE
#string=False makes it return the tuple (rmse,scc,pcc)
def formatMetrics(sol,key,string=True):
	distance_matrix=sol2distVec(sol)
	# key=np.nan_to_num(key,copy=True,posinf=0)
	distance_list=[]
	key_list=[]
	for i in range(len(distance_matrix)):
		for j in range(i+1):
			if key[i,j]==np.inf:
				continue
			distance_list.append(distance_matrix[i,j])
			key_list.append(key[i,j])
	pearson=scipy.stats.pearsonr(distance_list,key_list)[0]
	spearman=scipy.stats.spearmanr(distance_list,key_list).correlation
	rmse=sklearn.metrics.mean_squared_error(key_list,distance_list)**0.5
	if string:
		metrics=f"AVG RMSE: {rmse}\nAVG Spearman correlation Dist vs. Reconstructed Dist: {spearman}\nAVG Pearson correlation Dist vs. Reconstructed Dist: {pearson}"
	else:
		metrics=(rmse,spearman,pearson)
	return metrics
#this writes the log file
#accepts the metric string from formatMetrics,input file name, and an output file name
#writes .log file accordingly
def outputLog(metrics,alpha,input_fname,out_fname=None,bat_params=None,runtime=None,searched_alphas=None,structs=None):
	if out_fname is None:
		out_fname="bat.log"
	else:
		out_fname+=".log"
	outstring=f"Input file: {input_fname}\nConvert factor: {alpha}\n"+metrics
	if runtime is not None:
		outstring+=f"\nRuntime: {runtime:.2f} seconds"
	if searched_alphas is not None:
		outstring+=f"\nPerformed Alpha Search: {searched_alphas}"
	if structs is not None:
		outstring+=f"\nGenerated Stuctures Count: {structs}"
	if bat_params is not None:
		outstring+=f"\n"+"\n".join([f"{key}={value}" for key,value in bat_params.items()])
	f=open(out_fname,"w")
	f.write(outstring)
	f.close()
#prints the coordinate mapping to file
def outputCoordinateMap(index_dict,out_fname):
	if out_fname is None:
		out_fname="bat_coordinate_mapping.txt"
	else:
		out_fname+="coordinate_mapping.txt"
	outstring=""
	for key in index_dict:
		outstring+=f"{key}\t{index_dict[key]}\n"
	f=open(out_fname,"w")
	f.write(outstring)
	f.close()

#simple function for multiprocessing
def optimize(distance,params):
	np.random.seed(int(time.time()*100000000)%3200000000) #This is a nasty hack so linux will have differnt random seeds, else all structures produced are identical
	bats=Bat(distance,**params)
	return bats.fly()


#driver code
if __name__=="__main__":


	#command line/input processing
	start_time=time.time()
	parser=argparse.ArgumentParser()
	parser.add_argument("contact_matrix",help="File name of a white space delimited square contact matrix")
	parser.add_argument("params",help="File name of the parameters file")
	# parser.add_argument("parameters",default="parameters.txt",)	
	args=parser.parse_args()
	contact,alphas,perturbations,param_dict,outfile,structs=processInput(args.contact_matrix,args.params)

	print("Running ChromeBat")

	contact,index_dict=removeZeroRowCol(contact)#preprocesses away all zero row/col pairs

	#perform alpha search if no alpha found in the parameter file
	#sets alpha to best alpha found
	PROC_COUNT=cpu_count()
	searched_alphas=False

	if len(alphas) == 0:
		print("No Alpha values found... using alpha=[0.1,0.3,0.5,0.7,0.9,1]")
		alphas=[0.1,0.3,0.5,0.7,0.9,1]
	elif len(alphas)==1:
		alpha=alphas[0]

	if len(perturbations) == 0:
		print("No Perturbation Value found... using perturbation=[0.002,0.004,0.006,0.008,0.01]")
		perturbations=[0.002,0.004,0.006,0.008,0.01]
	elif len(alphas)==1:
		perturbation=perturbations[0]

	if len(alphas)*len(perturbations)>1:
		print(f"There are {len(alphas)*len(perturbations)} combinations of alpha and perturbation values")
		print(f"Performing Alpha/Perturbation Search...(this will open {len(alphas)*len(perturbations)} processes)")

		#multiprocessing to search the alpha values
		searched_alphas=True
		alpha_perturbations=list(product(alphas,perturbations))
		distance_m_list=[adjancenyPreprocess(contactToDistance(contact,a)) for a,p in alpha_perturbations]
		param_dict_list=[param_dict.copy() for i in range(len(alpha_perturbations))]
		for i in range(len(alpha_perturbations)):
			a,p=alpha_perturbations[i]
			param_dict_list[i]["perturbation"]=p
		
		# distance_m_list=[adjancenyPreprocess(contactToDistance(contact,a)) for a in alphas]
		# param_dict_list=[param]

		pool = Pool(processes=PROC_COUNT)
		swarms=pool.starmap(optimize, zip(distance_m_list,param_dict_list))
		pool.close()
		pool.join()

		#metrics computations and alpha selection
		spearmans=[formatMetrics(sol,distance_m_list[i],string=False)[1] for i,sol in enumerate(swarms)]
		best_index=np.argmax(spearmans)
		alpha,perturbation=alpha_perturbations[best_index]
		searched_sol=swarms[best_index]
		end_alphas_time=time.time()
		# pad_len=len(str(max(alphas,key=lambda x:len(str(x)))))+7
		# alpha_scc_string="\n".join([pad(alphas[i],pad_len,left=False)+str(spearmans[i]) for i in range(len(spearmans))])
		# print("Search Results:\n"+pad("alpha",pad_len)+"SCC\n"+alpha_scc_string)

		print("Search results in format (alpha,perturbation): SCC")
		for i in range(len(alpha_perturbations)):
			a,p=alpha_perturbations[i]
			print(f"({a},{p}): {spearmans[i]}")
		print(f"Best Alpha={alpha},perturbation={perturbation} found in {end_alphas_time-start_time:.2f} seconds")

	#geneterates structs structures 
	distance_m=contactToDistance(contact,alpha)
	param_dict["perturbation"]=perturbation
	if structs>0: 
		print(f"Generating {structs} more structures using alpha={alpha}, perturbation={perturbation}")
		distance_m_prepro=adjancenyPreprocess(distance_m)
		pool = Pool(processes=PROC_COUNT)
		swarms=pool.starmap(optimize, zip(repeat(distance_m_prepro,structs),repeat(param_dict)))
		pool.close()
		pool.join()
	else: # reset swarms variable if structs happened to be 0
		swarms=[]

	#metric computations on generated structures
	if searched_alphas: #reuse structure generated in the search portion
		swarms.append(searched_sol)

	if len(swarms)==0:
		raise ValueError("No structures generated, if structs=0 an alpha search must be performed to generate a structure")

	spearmans=[formatMetrics(sol,distance_m,string=False)[1] for sol in swarms]
	sorted_index=np.argsort(-np.array(spearmans)) #the negative makes it sort greatest to least

	final_end_time=time.time()

	print(f"Writing {structs} structures...")
	swarms=np.array(swarms)#needed for indexing in the next line
	for i,sol in enumerate(swarms[sorted_index]):
		sol*=10 # output structures tended to be very small so now they are not
		metrics=formatMetrics(sol,distance_m)
		base_name=outfile+str(i)
		outputPdb(sol,base_name)
		if i==0: #the 0th struct is best in terms of SCC and contains all run details
			print(f"The Best Structure's (structure 0) metrics are:")
			print(metrics)
			outputLog(metrics,alpha,args.contact_matrix,base_name,bat_params=param_dict,runtime=final_end_time-start_time,searched_alphas=searched_alphas,structs=structs)
			outputCoordinateMap(index_dict,base_name)
		else:
			outputLog(metrics,alpha,args.contact_matrix,base_name,bat_params=param_dict)

	print(f"{outfile}[0-{structs-1+searched_alphas}].log and {outfile}[0-{structs-1+searched_alphas}].pdb written!")
	print(f"Done in {final_end_time-start_time:.2f} seconds total")