import os
import sys
import argparse

def parse_args():
    """ parse input args """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="xml reports need to change name")
    parser.add_argument("--testsuite_old_name", type=str,
                        help="old testsuite name need to be changed")
    return parser.parse_args()

class Sed(object):
    """ Sed """
    def __init__(self, oldstr, newstr):
        """ __init__ """
        self.old_str = oldstr
        self.new_str = newstr

    def sedfile(self, old_file, tmp_file="tmp.xml"):
        """ sed file """
        with open(old_file, 'r') as self.f, open(tmp_file, "a+") as self.f1:
            for self.i in self.f:
                if self.old_str in self.i:
                    self.i = self.i.replace(self.old_str, self.new_str)
                self.f1.write(self.i)
                self.f1.flush()
        os.remove(old_file)  # delete old file
        os.rename(tmp_file, old_file)  # move tmp to origin

if __name__ == '__main__':
    args = parse_args()

    # e.g. test_AlexNet_gpu_1e-5_bz1.xml
    input_test_case_name = args.input_file.split('.xml')[0]

    # e.g. test_pdclas_model
    old_suite_name = args.testsuite_old_name

    sed = Sed(old_suite_name, input_test_case_name)
    sed.sedfile(args.input_file)
