# 参考：https://qiita.com/msrks/items/e264872efa062c7d6955

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def min_max_normalization(x):
    min = x.min()
    max = x.max()
    result = (x - min) / (max - min)
    return result


def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x - xmean) / xstd
    return zscore


def create_animation(file_name, score, is_random):
    if is_random:
        random_str = "_random"
    else:
        random_str = ""
    fig = plt.figure()
    joint_feature = np.load(file_name)
    for i in range(joint_feature.shape[0]):
        joint_feature[i] = min_max_normalization(joint_feature[i])
        # joint_feature[i] = zscore(joint_feature[i])

    joint_feature = joint_feature.transpose(1, 0, 2)
    x = np.arange(joint_feature.shape[1])
    y = np.arange(joint_feature.shape[2])
    ims = []
    min = joint_feature.min()
    max = joint_feature.max()

    for i in range(joint_feature.shape[0]):
        im = plt.imshow(joint_feature[i], vmin=min, vmax=max, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True,
                                    repeat_delay=1000)
    plt.colorbar()
    if "1st" in file_name:
        output_file_name = str(score) + "_1st_anim" + random_str
    elif "2nd" in file_name:
        output_file_name = str(score) + "_2nd_anim" + random_str
    elif "3rd" in file_name:
        output_file_name = str(score) + "_3rd_anim" + random_str
    else:
        output_file_name = "unknown"

    ani.save("./OutputAnimation/" + output_file_name + ".gif", writer="Pillow")
    ani.save("./OutputAnimation/" + output_file_name + ".mp4", writer="ffmpeg")
    plt.show()


# def distribution(file_name):
#     joint_feature = np.load(file_name)
#     for i in range(joint_feature.shape[0]):
#         min = joint_feature[i].min()
#         max= joint_feature.max()

def show_hist(file_name):
    joint_feature = np.load(file_name)[0]
    joint_feature=joint_feature.flatten()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)


    ax.hist(joint_feature,bins=500)
    ax.set_title('joint_feature  histgram')
    ax.set_xlabel('x')
    ax.set_ylabel('freq')
    fig.show()

if __name__ == '__main__':
    is_random =False
    if is_random:
        random_str = "_random"
    else:
        random_str = ""
    score = 20
    print("start!!")
    show_hist('./DataDir/joint_1st_' + str(score) + random_str + '.npy')
    # create_animation('./DataDir/joint_1st_' + str(score) + random_str + '.npy', score, is_random)
    # create_animation('./DataDir/joint_2nd_' + str(score) + random_str + '.npy', score, is_random)
    # create_animation('./DataDir/joint_3rd_' + str(score) + random_str + '.npy', score, is_random)
    print("end!!!")
