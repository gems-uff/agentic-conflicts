import pandas as pd
import random
import os
import time
import pylcs
from datetime import datetime
import sys
import argparse

# ==============================================================================
# --- GENERAL CONFIGURATIONS ---
# ==============================================================================
RANDOM_SEED = 3022024
random.seed(RANDOM_SEED)
DEBUG_ACTIVE = False
SAVE_CANDIDATE = True

# ==============================================================================
# --- ALGORITHM PARAMETERS (from tuning) ---
# ==============================================================================

# --- Best RRHC Configuration ---
RRHC_LOCAL_SEARCH_MAXIMUM_NEIGHBORS = 5
RRHC_ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT = 10
RRHC_ILS_TIMEOUT_SECONDS = 15

# --- Best GA Configuration ---
GA_POPULATION_SIZE = 200
GA_NUM_GENERATIONS = 150
GA_MUTATION_RATE = 0.1
GA_TOURNAMENT_SIZE = 2
GA_ELITISM_COUNT = 1
GA_TIMEOUT_SECONDS = 15 
GA_MAX_STAGNATION_ITERATIONS = 15
EVALUATION_FUNCTION = 'vanilla' # Best fitness function from tuning

# Internal timeouts (not tuned)
NEIGHBOR_FINDING_TIMEOUT_SECONDS = 3
LOCAL_SEARCH_TIMEOUT_SECONDS = 5
MAX_NEIGHBOR_TRYING = 200

# ==============================================================================
# --- SCRIPT ARGUMENTS AND FILE SETUP ---
# ==============================================================================
parser = argparse.ArgumentParser(description='Evaluation script for SBCR.')
parser.add_argument('--dataset', type=str, help='Dataset to be used (e.g., dataset1).')
parser.add_argument('--algorithm', type=str, default='ga', choices=['rrhc', 'ga'], help='Algorithm to evaluate.')
parser.add_argument('--configuration', type=str, default=None, help='RRHC configuration (e.g., "5,10,15"). Not used for GA.')
parser.add_argument('--function', type=str, default=None, choices=['vanilla', 'weighted'], help='Fuction to be used: vanilla or weighted.')
parser.add_argument('--execution_name', type=str, default=None, help='Name for the execution folder.')
args = parser.parse_args()

DATASET = args.dataset
ALGORITHM_TO_EVALUATE = args.algorithm
EXECUTION_NAME = args.execution_name if args.execution_name else f"{DATASET}_{ALGORITHM_TO_EVALUATE}"

# Override RRHC parameters if provided
if ALGORITHM_TO_EVALUATE == 'rrhc' and args.configuration:
    try:
        parts = args.configuration.split(',')
        RRHC_LOCAL_SEARCH_MAXIMUM_NEIGHBORS = int(parts[0])
        RRHC_ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT = int(parts[1])
        RRHC_ILS_TIMEOUT_SECONDS = int(parts[2])
        print(f"Using provided RRHC Configuration: {RRHC_LOCAL_SEARCH_MAXIMUM_NEIGHBORS}, {RRHC_ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT}, {RRHC_ILS_TIMEOUT_SECONDS}")
    except:
        print("Invalid RRHC configuration format. Using defaults.")

if args.function:
    EVALUATION_FUNCTION = args.function
        

# FOLDER SETUP
# Ensure we are saving results in a 'results' directory to avoid cluttering root
RESULTS_DIR_BASE = "results"
EXECUTION_FOLDER = os.path.join(RESULTS_DIR_BASE, EXECUTION_NAME)

OUTPUT_FOLDER = os.path.join(EXECUTION_FOLDER, "OUTPUT")
RESULTS_FOLDER = os.path.join(EXECUTION_FOLDER, "RESULTS")
TMP_FOLDER = os.path.join(EXECUTION_FOLDER, "tmp")

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TMP_FOLDER, exist_ok=True)
RESULT_FILE = os.path.join(RESULTS_FOLDER, f'results_evaluate_{DATASET}_{EXECUTION_NAME}_seed-{RANDOM_SEED}.xlsx')


# ==============================================================================
# --- SHARED FUNCTIONS ---
# ==============================================================================

def get_candidate_text(indexes, v1, v2):
    candidate = []
    v1_lines = v1.splitlines()
    v2_lines = v2.splitlines()
    for side, index in indexes:
        if side == 'v1' and index < len(v1_lines):
            candidate.append(v1_lines[index])
        elif side == 'v2' and index < len(v2_lines):
            candidate.append(v2_lines[index])
    return '\n'.join(candidate)

"""
    Given a list, returns `n` sorted random indexes from it
"""
def get_sorted_random_indexes(original_list, n):
    # Get n random indices
    random_indices = random.sample(range(len(original_list)), n)

    # Sort the indices in ascending order
    sorted_indices = sorted(random_indices)

    return sorted_indices

"""
    Given two source code versions, returns a random candidate,
    respecting the partial order of the versions
    Uses the candidate data structure: [('v1', 0), ('v1', 1), ('v1', 2), ('v2', 1)]
"""
def partial_order_random_candidate(v1, v2):
    v1_lines = v1.splitlines()
    v2_lines = v2.splitlines()
    number_lines_v1 = random.randrange(len(v1_lines)+1)
    number_lines_v2 = random.randrange(len(v2_lines)+1)
    candidate_size = number_lines_v1 + number_lines_v2
    
    indexes_v1 = get_sorted_random_indexes(v1_lines, number_lines_v1)
    indexes_v2 = get_sorted_random_indexes(v2_lines, number_lines_v2)
    
    candidate = []
    while(len(candidate) < candidate_size):
        line_source = ''
        if(len(indexes_v1)>0 and len(indexes_v2)>0):
            line_source = random.choice(['v1', 'v2'])
        elif len(indexes_v2)==0:
            line_source='v1'
        else:
            line_source='v2'
        if line_source == 'v1':
            candidate.append(('v1', indexes_v1.pop(0)))
        else:
            candidate.append(('v2', indexes_v2.pop(0)))

    candidate_text = get_candidate_text(candidate, v1, v2)
    # we dont allow resolutions equal to v1 or v2 (only combination)
    if candidate_text == v1 or candidate_text == v2:
        return partial_order_random_candidate(v1, v2)
    return candidate

'''
    A resolution violates the partial order when there is no way to arrange the 
        chunk lines to compose the resolution without breaking their original order
    Uses candidate data structure, thus the analysis is based on matching lines indexes
'''
def candidate_has_partial_order(candidate):
    
    last_v1_index = -1
    last_v2_index = -1
    for candidate_element in candidate:
        line_side = candidate_element[0]
        line_index = candidate_element[1]
        if line_side == 'v1':
            if line_index < last_v1_index:
                return False
            last_v1_index = line_index
        else:
            if line_index < last_v2_index:
                return False
            last_v2_index = line_index
    return True
    
def write_file(content, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(str(content))

def get_gestalt_from_text(text1, text2):
    try:
        if not text1 and not text2: return 1.0
        if not text1 or not text2: return 0.0
        if text1 == text2: return 1.0
        return 2 * pylcs.lcs(text1, text2) / (len(text1) + len(text2))
    except Exception:
        return 0.0

def evaluate_vanilla(candidate_text, v1, v2):
    v1_similarity = get_gestalt_from_text(candidate_text, v1)
    v2_similarity = get_gestalt_from_text(candidate_text, v2)
    return (v1_similarity + v2_similarity) / 2

def evaluate_weighted(candidate_text, v1, v2):
    total_size = len(v1) + len(v2)
    if total_size == 0: return 1.0
    v1_similarity = get_gestalt_from_text(candidate_text, v1)
    v2_similarity = get_gestalt_from_text(candidate_text, v2)
    return (v1_similarity * len(v1) + v2_similarity * len(v2)) / total_size

def evaluate(candidate_text, v1, v2):
    if EVALUATION_FUNCTION == 'weighted':
        return evaluate_weighted(candidate_text, v1, v2)
    else: # vanilla
        return evaluate_vanilla(candidate_text, v1, v2)

def compare(content1, content2):
    return get_gestalt_from_text(content1, content2)

"""
    Given a candidate, a position and which side to look for (v1 or v2), 
    returns the first index before the position that is from the side
"""
def get_first_index_before_position(candidate, position, side):
    first_index_before = -1
    if position > 0:
        for element in candidate[position-1::-1]:
            element_side = element[0]
            element_index = element[1]
            if element_side == side:
                first_index_before = element_index
                break
    return first_index_before

"""
    Given a candidate, a position and which side to look for (v1 or v2),
    returns the first index after the position that is from the side
"""
def get_first_index_after_position(candidate, position, side):
    first_index_after = float('inf')
    if position < len(candidate):
        for element in candidate[position:]:
            element_side = element[0]
            element_index = element[1]
            if element_side == side:
                first_index_after = element_index
                break
    return first_index_after

'''
    Given a candidate, a position and the two versions,
    returns the lines that can be added to the candidate at the position,
    respecting the partial order
        Candidate example: [('v1', 0), ('v1', 1), ('v1', 2), ('v2', 1)]
'''
def find_feasible_lines_to_add(candidate, v1, v2, position):
    feasible_lines = []
    # Get the lines that are not in the candidate
    v1_lines_indexes = list(range(len(v1.splitlines())))
    v2_lines_indexes = list(range(len(v2.splitlines())))
    for candidate_element in candidate:
        line_side = candidate_element[0]
        line_index = candidate_element[1]
        v1_lines_indexes.remove(line_index) if line_side == 'v1' else v2_lines_indexes.remove(line_index)
    
    # Get the lines that can be added to the candidate
    v1_index_before = get_first_index_before_position(candidate, position, 'v1')
    v2_index_before = get_first_index_before_position(candidate, position, 'v2')
    v1_index_after = get_first_index_after_position(candidate, position, 'v1')
    v2_index_after = get_first_index_after_position(candidate, position, 'v2')

    for line_index in v1_lines_indexes:
        if line_index > v1_index_before and line_index < v1_index_after:
            feasible_lines.append(('v1', line_index))

    for line_index in v2_lines_indexes:
        if line_index > v2_index_before and line_index < v2_index_after:
            feasible_lines.append(('v2', line_index))
    
    return feasible_lines

"""
    Given a candidate and a side (v1 or v2),
    returns the index of the last element from the side in the candidate
"""
def get_last_side_index(candidate, side):
    last_side_index = -1
    for candidate_element in candidate[-1::-1]:
        element_side = candidate_element[0]
        if element_side == side:
            last_side_index = candidate_element[1]
            break
    return last_side_index

'''
    Given a candidate, returns a random neighbor
        A neighbor is defined as a candidate that differs
        at most in one line (either by addition, removal or swapping)
        Candidate example: [('v1', 0), ('v1', 1), ('v1', 2), ('v2', 1)]
'''
def get_random_neighbor(candidate, v1, v2):
    neighbor = candidate.copy()
    v1_lines = v1.splitlines()
    v2_lines = v2.splitlines()
    feasible_actions = []
    if len(candidate) > 0:
        random_position = random.randint(0, len(candidate)-1)
        feasible_actions.append('remove')
        if random_position > 0:
                # we can only swap when the other element is from the opposite side
                if candidate[random_position-1][0] != candidate[random_position][0]:
                    feasible_actions.append('swap_before')
        if random_position < len(candidate)-1:
            # we can only swap when the other element is from the opposite side
            if candidate[random_position+1][0] != candidate[random_position][0]:
                feasible_actions.append('swap_after')
        if len(candidate) < len(v1_lines) + len(v2_lines):
            feasible_lines = find_feasible_lines_to_add(candidate, v1, v2, random_position)
            if len(feasible_lines) > 0:
                feasible_actions.append('add')
    else: # candidate is empty, only add is possible
        random_position = random.randint(0, len(candidate))
        feasible_actions = ['add']
        feasible_lines = find_feasible_lines_to_add(candidate, v1, v2, random_position)
    raffled_action = random.choice(feasible_actions)
    
    if raffled_action == 'remove':
        neighbor.pop(random_position)
    elif raffled_action == 'swap_before':
        neighbor[random_position], neighbor[random_position-1] = neighbor[random_position-1], neighbor[random_position]
    elif raffled_action == 'swap_after':
        neighbor[random_position], neighbor[random_position+1] = neighbor[random_position+1], neighbor[random_position]
    else: # add
        if random_position == len(candidate)-1: # a chance to add at the end
            if random.choice([0,1]) == 1:
                # we can only add at the end when either the last v1 index or last v2 index 
                # are smaller than the length of v1 and v2, respectively
                last_v1_index = get_last_side_index(candidate, 'v1')
                last_v2_index = get_last_side_index(candidate, 'v2')
                if last_v1_index < len(v1.splitlines())-1 or last_v2_index < len(v2.splitlines())-1:
                    random_position += 1
                    # need to update the feasible lines beause the position has changed
                    feasible_lines = find_feasible_lines_to_add(candidate, v1, v2, random_position)
        if len(feasible_lines) > 0:
            random_line = random.choice(feasible_lines)
            neighbor.insert(random_position, random_line)
        else:
            print('No feasible lines to add')
            print('candidate', candidate, 'position:', random_position)
    # print('raffled_action', raffled_action, 'random_position', random_position)
    return neighbor


# ==============================================================================
# --- RANDOM RESTART HILL CLIMBING (RRHC) ALGORITHM ---
# ==============================================================================

'''
    Get a random neighbor and 
        check if the neighbor satisfies the partial order,
        Otherwise, generate another, until one that satisfies is found
    Timeouts if no partial order complaint neighbor is found
'''
def get_next_neighbor(candidate, v1, v2, neighbors):
    random_neighbor = get_random_neighbor(candidate, v1, v2)
    random_neighbor_text = get_candidate_text(random_neighbor, v1, v2)
    start_time = time.time()
    count_tries = 0
    has_partial_order = candidate_has_partial_order(random_neighbor)
    # we dont allow resolutions equal to v1 or v2 (only combination)
    while not has_partial_order or hash(random_neighbor_text) in neighbors or random_neighbor_text == v1 or random_neighbor_text == v2:
        if time.time() - start_time > NEIGHBOR_FINDING_TIMEOUT_SECONDS or count_tries >= MAX_NEIGHBOR_TRYING:
            return None
        random_neighbor = get_random_neighbor(candidate, v1, v2)
        random_neighbor_text = get_candidate_text(random_neighbor, v1, v2)
        count_tries+=1
        has_partial_order = candidate_has_partial_order(random_neighbor)
        if not has_partial_order:
            print('Violates partial order', flush=True)
    return random_neighbor

def get_fittest(neighbors):
    fittest_neighbor = None
    fittest_value = -9999999
    for neighbor in neighbors.items():
        if neighbor[1][1] > fittest_value:
            fittest_neighbor = neighbor[1][0]
            fittest_value = neighbor[1][1]
    return fittest_neighbor, fittest_value

def local_search(starting_candidate, starting_candidate_fitness, n, v1, v2, source, depth):
    neighbors = {}
    start_time = time.time()
    tries_count = 0
    while len(neighbors) < n and time.time() - start_time < LOCAL_SEARCH_TIMEOUT_SECONDS and tries_count < MAX_NEIGHBOR_TRYING:
        s_new = get_next_neighbor(starting_candidate, v1, v2, neighbors)
        if s_new != None:
            f_new = evaluate(get_candidate_text(s_new, v1, v2), v1, v2)
            neighbors[hash(get_candidate_text(s_new, v1, v2))] = [s_new, f_new]
        else:
            tries_count+=1

    s_candidate, f_candidate = get_fittest(neighbors)
    if f_candidate > starting_candidate_fitness:
        return local_search(s_candidate, f_candidate, n, v1, v2, 'ls_in', depth+1)
    else:
        return starting_candidate, starting_candidate_fitness

def pertubate(candidate, v1, v2):
    return partial_order_random_candidate(v1, v2)

def rrhc_resolution(v1, v2):

    s_star = partial_order_random_candidate(v1, v2)
    f_star = evaluate(get_candidate_text(s_star, v1, v2), v1, v2)

    s_star, f_star = local_search(s_star, f_star, RRHC_LOCAL_SEARCH_MAXIMUM_NEIGHBORS, v1, v2, 'initial', 0)

    start_time = time.time()
    iteration_number = 1
    iterations_without_improvement = 0
    while (time.time() - start_time < RRHC_ILS_TIMEOUT_SECONDS) and iterations_without_improvement <= RRHC_ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT:
        s_new = pertubate(s_star, v1, v2)
        f_new = evaluate(get_candidate_text(s_new, v1, v2), v1, v2)


        s_star_new, f_star_new = local_search(s_new, f_new, RRHC_LOCAL_SEARCH_MAXIMUM_NEIGHBORS, v1, v2, f'it-{iteration_number}', 0)

        if f_star_new > f_star:
            s_star = s_star_new
            f_star = f_star_new
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
        iteration_number+=1
    return s_star, f_star 

# ==============================================================================
# --- GENETIC ALGORITHM (GA) ---
# ==============================================================================

def selection_ga(ranked_population):
    """ Tournament Selection """
    tournament = random.sample(ranked_population, GA_TOURNAMENT_SIZE)
    winner = max(tournament, key=lambda item: item[1])
    return winner[0]

def crossover_ga(parent1, parent2):
    """
    Performs a crossover based on the union of parents' genes (gene pool).
    This prevents duplicate genes AND avoids a bias towards larger children.
    """
    if not parent1 and not parent2:
        return [], []

    # Step 1: Create the gene pool with unique genes from both parents.
    gene_pool = list(set(parent1) | set(parent2))
    
    children = []
    for _ in range(2): # Create two children
        # Step 2: Determine a variable size for the child to avoid size bias.
        min_size = min(len(parent1), len(parent2))
        max_size = len(gene_pool)
        if min_size == 0 and max_size == 0:
            children.append([])
            continue

        child_size = random.randint(min_size, max_size) if min_size < max_size else min_size

        # Step 3: Randomly sample genes from the pool to form the child's content.
        child_content = random.sample(gene_pool, child_size)

        # Step 4: Enforce partial order on the sampled genes.
        # Separate genes by source (v1, v2) and sort them by their original index.
        v1_genes = sorted([g for g in child_content if g[0] == 'v1'], key=lambda x: x[1])
        v2_genes = sorted([g for g in child_content if g[0] == 'v2'], key=lambda x: x[1])

        # Merge them back together randomly, preserving the now-sorted partial order.
        ordered_child = []
        while v1_genes or v2_genes:
            source_choice = []
            if v1_genes: source_choice.append('v1')
            if v2_genes: source_choice.append('v2')
            
            chosen_source = random.choice(source_choice)
            
            if chosen_source == 'v1':
                ordered_child.append(v1_genes.pop(0))
            else:
                ordered_child.append(v2_genes.pop(0))
        
        children.append(ordered_child)

    return children[0], children[1]

def genetic_algorithm_resolution(v1, v2):
    # 1. Initialization
    population = [partial_order_random_candidate(v1, v2) for _ in range(GA_POPULATION_SIZE)]
    
    best_solution = None
    best_fitness = -1
    stagnation_iterations = 0
    start_time = time.time()
    for generation in range(GA_NUM_GENERATIONS):
        if time.time() - start_time > GA_TIMEOUT_SECONDS:
            # print("GA limit time.", file=sys.stderr)
            break
        if stagnation_iterations >= GA_MAX_STAGNATION_ITERATIONS:
            # print("GA stagnated.", file=sys.stderr)
            break

        # 2. Evaluation
        ranked_population = []
        for individual in population:
            text = get_candidate_text(individual, v1, v2)
            fitness = evaluate(text, v1, v2)
            ranked_population.append((individual, fitness))
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual
                stagnation_iterations = 0
            else:
                stagnation_iterations+=1

        ranked_population.sort(key=lambda item: item[1], reverse=True)
        debug_message(f"Generation # {generation+1}: Best Fitness = {best_fitness:.4f}")

        # 3. Selection and reproduction
        new_population = [ranked_population[i][0] for i in range(GA_ELITISM_COUNT)] # Elitism: keep the best individuals

        while len(new_population) < GA_POPULATION_SIZE:
            parent1 = selection_ga(ranked_population)
            parent2 = selection_ga(ranked_population)
            
            # 4. Crossover
            child1, child2 = crossover_ga(parent1, parent2)
            
            # 5. Mutation
            if random.random() < GA_MUTATION_RATE:
                child1 = get_random_neighbor(child1, v1, v2)
                # Assure the mutation didnt violate partial order
                if not candidate_has_partial_order(child1):
                    child1 = parent1  # Reverts if invalid mutation

            if random.random() < GA_MUTATION_RATE:
                child2 = get_random_neighbor(child2, v1, v2)
                if not candidate_has_partial_order(child2):
                    child2 = parent2 # Reverts if invalid mutation

            new_population.append(child1)
            if len(new_population) < GA_POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_solution, best_fitness

def debug_message(msg):
    if DEBUG_ACTIVE:
        print(msg)

# ==============================================================================
# --- MAIN EXPERIMENT FUNCTION ---
# ==============================================================================

def adapt_dataset(df):
    if 'chunk_id' not in df.columns:
        df['chunk_id'] = df['merge_id'].astype(str) + '-' + df['chunk_number'].astype(str)
    if 'v1' not in df.columns:
        df.rename(columns={'all_raw_a':'v1', 'all_raw_b':'v2', 'all_raw_res':'solution'}, inplace=True)


def execute_experiment():
    df_chunks = pd.read_json(f"data/{DATASET}_testing.json")
    df_chunks = df_chunks.sample(frac=1, random_state=RANDOM_SEED)
    data = []
    adapt_dataset(df_chunks)
    # Define columns based on algorithm
    if ALGORITHM_TO_EVALUATE == 'rrhc':
        columns = ['chunk_id', 'fitness', 'solution_similarity', 'status', 'time_seconds', 
                   'max_neighbors', 'max_stagnation', 'timeout']
    else: # ga
        columns = ['chunk_id', 'fitness', 'solution_similarity', 'status', 'time_seconds',
                   'population_size', 'num_generations', 'mutation_rate', 'tournament_size', 
                   'elitism_count', 'max_stagnation', 'fitness_function']

    chunk_count = 0
    for index, row in df_chunks.iterrows():
        chunk_count += 1
        chunk_id = row['chunk_id']
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ### Analyzing chunk {chunk_count}/{len(df_chunks)} ({chunk_id})", flush=True)

        v1 = row.get('v1', row.get('all_raw_a', ''))
        v2 = row.get('v2', row.get('all_raw_b', ''))
        solution = row.get('solution', row.get('all_raw_res', ''))

        if len(v1) > 20000 or len(v2) > 20000:
            print(f"Skipping chunk {chunk_id} due to size limits.", flush=True)
            data.append([chunk_id, None, None, "OUT_OF_LIMITS", None] + [None]*(len(columns)-5))
            continue

        start_time = time.time()
        candidate, fitness = None, -1.0

        try:
            if ALGORITHM_TO_EVALUATE == 'rrhc':
                candidate, fitness = rrhc_resolution(v1, v2)
            else: # ga
                candidate, fitness = genetic_algorithm_resolution(v1, v2)

            elapsed_time = time.time() - start_time
            
            if candidate:
                candidate_text = get_candidate_text(candidate, v1, v2)
                solution_similarity = compare(solution, candidate_text)
                if SAVE_CANDIDATE:
                    write_file(candidate_text, f"{OUTPUT_FOLDER}/{chunk_id}")
            else:
                solution_similarity = 0

            # Append data according to columns
            if ALGORITHM_TO_EVALUATE == 'rrhc':
                data.append([chunk_id, fitness, solution_similarity, 'ok', elapsed_time,
                             RRHC_LOCAL_SEARCH_MAXIMUM_NEIGHBORS, RRHC_ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT, RRHC_ILS_TIMEOUT_SECONDS])
            else: # ga
                data.append([chunk_id, fitness, solution_similarity, 'ok', elapsed_time,
                             GA_POPULATION_SIZE, GA_NUM_GENERATIONS, GA_MUTATION_RATE, GA_TOURNAMENT_SIZE,
                             GA_ELITISM_COUNT, GA_MAX_STAGNATION_ITERATIONS, EVALUATION_FUNCTION])
        
        except Exception as e:
            print(f"ERROR processing chunk {chunk_id}: {e}", flush=True)
            data.append([chunk_id, None, None, str(e), None] + [None]*(len(columns)-5))

    pd.DataFrame(data, columns=columns).to_excel(RESULT_FILE, index=False)
    print(f"\n--- Experiment finished. Results saved to: {RESULT_FILE} ---")

if __name__ == "__main__":
    execute_experiment()