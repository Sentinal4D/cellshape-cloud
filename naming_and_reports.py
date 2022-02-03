import fnmatch
import os


# Define printing to console and file
def print_both(f, text):
    print(text)
    f.write(text + '\n')


def reports(model_name, output_dir):
    dirs = [output_dir + 'runs', output_dir + 'reports', output_dir + 'nets']
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))
    # Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected
    reports_list = sorted(os.listdir(output_dir + 'reports'), reverse=True)
    if reports_list:
        for file in reports_list:
            # print(file)
            if fnmatch.fnmatch(file, model_name + '*'):
                idx = int(str(file)[-7:-4]) + 1
                break
    try:
        idx
    except NameError:
        idx = 1

    name = model_name + '_' + str(idx).zfill(3)
    # Filenames for report and weights
    name_txt = name + '.txt'
    name_net = name
    pretrained = name + '_pretrained.pt'
    # Arrange filenames for report, network weights, pretrained network weights
    name_txt = os.path.join(output_dir + 'reports', name_txt)
    name_net = os.path.join(output_dir + 'nets', name_net)
    pretrained = os.path.join(output_dir + 'nets', pretrained)
    f = open(name_txt, 'w')

    return f, name_net, pretrained, name_txt, name
