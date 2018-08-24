import time

class IO(object):
    def get_time(self):
        ct  = time.time()
        local_time = time.localtime(ct)
        data_head = time.strftime('%Y-%m-%d %H:%M:%S',local_time)
        data_secs = (ct-long(ct))*1000
        time_stamp = '%s.%03d' % (data_head,data_secs)
        return time_stamp

    def log_in(self,log_pos,log_posb,fun,i,epsize,pos,str):
        if (fun == 1):
            with open(log_pos,'a+') as f:
                f.write('-------------epoch:%d/%d-------------\n'%(i,epsize))

        if (fun == 2):
            with open(log_pos,'a+') as f:
                f.write(self.get_time())
                f.write('\nStart(%f,%f),  Goal(%f,%f)\n'%(pos[0],pos[1],pos[2],pos[3]))

        if (fun == 3):
             with open(log_pos,'a+') as f:
                 f.write(str)
                 f.write('\n\n')
             with open(log_posb,'a+') as f:
                 f.write(str)
                 f.write('\n\n')

        if (fun == 4):
             with open(log_posb,'a+') as f:
                 str = '\tFail:  Start(%f,%f),  Goal(%f,%f)\n'%(pos[0],pos[1],pos[2],pos[3])
                 f.write(str)


