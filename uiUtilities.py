from  data_process_train import parser, mapping, deeplog_sampling, generate_train_test
import os
def ui_train(input_dir='./trainInput/', output_dir='./trainOutput/', log_file="nova-sample-training.log", log_format='<Level> <Component> <ADDR> <Content>'):

    print("in ui_train, input_dir:",str(input_dir))
    parseResult = parser(input_dir, output_dir, log_file,log_format)
    # print("in ui_train:",str(output_dir) + "\event_sequence.csv")
    trainResult = generate_train_test(str(output_dir) + "\event_sequence.csv")
    return {parseResult, trainResult}

def ui_predict():
    print("hello world")
    return


if __name__ == "__main__":
    # ui_train()
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))