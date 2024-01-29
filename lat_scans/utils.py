import os
import json
import pandas as pd
import numpy as np
import random
from transformers import PreTrainedTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap


def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
            untruthful_statements.append(f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement)

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }

def plot_detection_results(input_ids, rep_reader_scores_dict, ax, index, start_answer_token=":", THRESHOLD=0.0):
    cmap = LinearSegmentedColormap.from_list('rg', ["r", (255/255, 255/255, 224/255), "g"], N=256)
    colormap = cmap

    # Define words and their colors
    words = [token.replace('▁', ' ') for token in input_ids]

    start_tok = words.index('ĠASS')
    cleaned_words = [word.replace('Ġ', ' ') for word in words]

    # Set limits for the x and y axes
    xlim = 1000
    ax.set_xlim(0, xlim)
    ylim = 20
    ax.set_ylim(0, ylim)

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Initialize positions and other parameters
    x_start, y_start = 1, 8
    x, y = x_start, y_start
    max_line_width = xlim
    word_width = 0
    iter = 0

    selected_concepts = ["honesty"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):

        rep_scores = np.array(rep_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean+5*std) | (rep_scores < mean-5*std)] = mean # get rid of outliers
        mag = max(0.3, np.abs(rep_scores).std() / 10)
        min_val, max_val = -mag, mag
        norm = Normalize(vmin=min_val, vmax=max_val)

        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD # change this for threshold
            rep_scores = rep_scores / np.std(rep_scores[start_tok:])
            rep_scores = np.clip(rep_scores, -mag, mag)
        if "flip" in n_style:
            rep_scores = -rep_scores
        
        rep_scores[np.abs(rep_scores) < 0.0] = 0

        # ofs = 0
        # rep_scores = np.array([rep_scores[max(0, i-ofs):min(len(rep_scores), i+ofs)].mean() for i in range(len(rep_scores))]) # add smoothing
        
        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)


        # Initialize positions and maximum line width
        x, y = x_start, y_start
        max_line_width = xlim
        started = False

        # Use the renderer associated with the Axes' figure
        renderer = ax.get_figure().canvas.get_renderer()
        for word, score in zip(cleaned_words[start_tok:], rep_scores[start_tok:]):

            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue
            score = score[0]
            color = colormap(norm(score))

            # Check if the current word would exceed the maximum line width
            if x + word_width > max_line_width:
                # Move to next line
                x = x_start
                y -= 3

            # Compute the width of the current word using the renderer
            text = ax.text(x, y, word, fontsize=13)
            text_bbox = text.get_window_extent(renderer=renderer)
            word_width = text_bbox.transformed(ax.transData.inverted()).width
            word_height = text_bbox.transformed(ax.transData.inverted()).height

            # Remove the previous text
            if iter:
                text.remove()
            
            y_pad = 1.3

            # Add the text with background color
            text = ax.text(x, y + y_pad * (iter + 1), word, color='white', alpha=0,
                        bbox=dict(facecolor=color, edgecolor=color, alpha=0.8, boxstyle=f'round,pad=0', linewidth=0),
                        fontsize=13)
            
            # Update the x position for the next word
            x += word_width + 0.1
        
        iter += 1

def plot_lat_scans(input_ids_dict, rep_reader_scores_dict, ax, index, model_title, input_labels):

    scores = rep_reader_scores_dict['honesty']
    start_tok = input_ids_dict['input'].index('ĠASS')
    end_index = min(start_tok + 60, len(scores))
    standardized_scores = np.array(scores)[start_tok:end_index, 20:]

    bound = 2.3
    threshold = 0
    standardized_scores[np.abs(standardized_scores) < threshold] = 1
    standardized_scores = standardized_scores.clip(-bound, bound)

    cmap = 'coolwarm'

    sns.heatmap(-standardized_scores.T, cmap=cmap, linewidth=0.5, annot=False, fmt=".3f", vmin=-bound, vmax=bound, ax=ax)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(0, len(standardized_scores), 5)[1:])
    ax.set_xticklabels(np.arange(0, len(standardized_scores), 5)[1:])
    ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
    ax.set_yticklabels(np.arange(20, len(standardized_scores[0])+20, 5)[::-1][1:])
    if input_labels[index] == 1:
        ax.set_title(f"[{index}] DEPLOYMENT LAT {model_title}")
    else:
        ax.set_title(f"[{index}] TRAINING LAT {model_title}")

def create_subplot(input_ids_dict, rep_reader_scores_dict, chosen_idxs, file_name, plot_func, **kwargs):
    # Create a directory to save the files
    directory = file_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    num_plots = len(chosen_idxs)
    if plot_func == plot_detection_results:
        fig, axs = plt.subplots(1, num_plots, figsize=(4*num_plots, 6))
    else:
        fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))

    for i, idx in enumerate(chosen_idxs):
        plot_func(input_ids_dict[idx], rep_reader_scores_dict[idx], axs[i], i, **kwargs)
        if i > 0:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])

    plt.tight_layout()
    plt.savefig(os.path.join(directory, file_name + "_lat.png"))  # Save the figure in the created directory

    # Save input IDs to a text file
    with open(os.path.join(directory, "input_ids_dict.json"), "w") as file:
        json.dump(input_ids_dict, file, indent=4)

    # Save rep_reader_scores_dict as a JSON file
    with open(os.path.join(directory, "rep_reader_scores_dict.json"), "w") as file:
        json.dump(rep_reader_scores_dict, file, indent=4)