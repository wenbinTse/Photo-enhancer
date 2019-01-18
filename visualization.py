import re
from matplotlib import pyplot as plt

names = [
    'disc_acc_train', 'disc_acc_test', 'gene_loss_train', 'gene_loss_test', 'content', 'color', 'texture', 'tv', 'psnr', 'ssim'
]


def show(path, all=True):
    file = open('./history/{}/iphone.txt'.format(path), 'r')
    format_file = open('./history/{}/format.csv'.format(path), 'w')

    vals = [[] for _ in range(10)]
    steps = []

    content = file.read()
    nums = [ float(i) for i in re.findall(r'-?\d+\.?\d*e?-?\d*', content) ]

    for idx, val in enumerate(nums):
        if idx % 11 == 0:
            steps.append(int(val))
        else:
            vals[idx % 11 - 1].append(val)

    print(steps)

    # for idx, val in enumerate(vals):
    #     if all:
    #         plt.subplot(5, 2, idx+1)
    #         plt.scatter(steps, val)
    #         plt.plot(steps,val, label=names[idx])
    #         plt.legend()
    #     else:
    #         plt.figure(figsize=(15, 7))
    #         plt.scatter(steps, val)
    #         for i, v in enumerate(val):
    #             if i % 3 == 0:
    #                 plt.annotate(v, (steps[i], v))
    #         plt.plot(steps, val, label=names[idx])
    #         plt.legend()
    #         plt.savefig('{}_{}.jpg'.format(path.split('/')[0], names[idx]))
    #         plt.close()

    if all:
        plt.show()

    ###########################
    # save to csv
    title = 'step'
    for name in names:
        title += ',' + name
    format_file.write(title + '\n')
    for idx, step in enumerate(steps):
        tmp = str(step)
        for i in range(10):
            tmp += ',' + str(vals[i][idx])
        tmp += '\n'
        format_file.write(tmp)


if __name__ == '__main__':
    show('/wgan_low_weight/models', False)