#%%
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

record = np.load('./saved_model/RLFM_uniform4record_s2_train_record.npz')
loss_record = record['loss_record']
in_prob_record = record['in_prob_record']


#%%
plt.plot(loss_record)
plt.grid()
plt.xlabel('x100 iterations')
plt.ylabel('loss')
plt.show()

plt.plot(in_prob_record)
plt.grid()
plt.xlabel('x100 iterations')
plt.ylabel('Probability of satisfying the constraint')
plt.show()