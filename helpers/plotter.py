import matplotlib.pyplot as plt


def plot(ident, filename, scores, total_no_records, global_reward):
    plt.title('Model Scores')
    plt.plot(scores, label="Score", color='b')
    # max_score = total_no_records * global_reward
    # plt.axhline(y=max_score, color='r', linestyle='-')
    plt.ylabel('Reward')
    plt.xlabel('Training Round')
    file_name = "history/figures/" + filename + "/fig" + str(ident) + ".png"
    plt.savefig(file_name)
