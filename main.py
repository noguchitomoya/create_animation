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


def create_animation(file_name, score):
    fig = plt.figure()
    joint_feature = np.load(file_name)
    for i in range(joint_feature.shape[0]):
        # joint_feature[i] = min_max_normalization(joint_feature[i])
        joint_feature[i] = zscore(joint_feature[i])


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
        output_file_name = str(score) + "_1st_anim"
    elif "2nd" in file_name:
        output_file_name = str(score) + "_2nd_anim"
    elif "3rd" in file_name:
        output_file_name = str(score) + "_3rd_anim"
    else:
        output_file_name = "unknown"

    ani.save(output_file_name + ".gif", writer="Pillow")
    ani.save(output_file_name + ".mp4", writer="ffmpeg")
    plt.show()


if __name__ == '__main__':
    score = 30
    print("start!!")
    create_animation('./DataDir/joint_1st_' + str(score) + '.npy', score)
    create_animation('./DataDir/joint_2nd_' + str(score) + '.npy', score)
    create_animation('./DataDir/joint_3rd_' + str(score) + '.npy', score)
    print("end!!!")
