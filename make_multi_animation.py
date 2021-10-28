import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_movie(show_fig=True, save_mp4=False, save_gif=False):
    fig = plt.figure()
    ax = []
    ims = []

    joint_feature_1st = np.load("joint_1st.npy")
    joint_feature_1st = joint_feature_1st.transpose(1, 0, 2)
    joint_feature_2nd = np.load("joint_2nd.npy")
    joint_feature_2nd = joint_feature_2nd.transpose(1, 0, 2)
    joint_feature_3rd = np.load("joint_3rd.npy")
    joint_feature_3rd = joint_feature_3rd.transpose(1, 0, 2)

    for i in range(3):
        ax.append(fig.add_subplot(3, 1, i + 1))

    x = np.arange(joint_feature_1st.shape[1])
    y = np.arange(joint_feature_1st.shape[2])
    # y = np.arange(0, 10, 0.1)
    # t = np.arange(0, 20, 1)
    # mx = np.arange(0, 10)
    # my = np.arange(0, 10)
    # X, Y = np.meshgrid(mx, my)

    first = True

    for i in range(joint_feature_1st.shape[0]):
        ax[0]=plt.imshow(joint_feature_1st[i], animated=True)
        im=ax[0]

        ax[1]=plt.imshow(joint_feature_2nd[i], animated=True)


        ims.append(im)
        first = False

    ani = animation.ArtistAnimation(fig, ims, interval=100)

    if show_fig is True:
        plt.show()
    if save_mp4 is True:
        # mp4での保存にはffpmegのパスが通っている必要がある
        ani.save("plot.mp4", writer="ffmpeg", dpi=300)
    if save_gif is True:
        # gifでの保存にはmatplotlibにimagemagickのパスが通っている必要がある
        ani.save("plot.gif", writer="imagemagick")


if __name__ == "__main__":
    # print(matplotlib.matplotlib_fname())  # matplotlibの設定ファイル確認用
    create_movie(False, True, True)
