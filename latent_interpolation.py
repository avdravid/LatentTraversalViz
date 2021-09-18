delta = 0.1

fig=plt.figure(figsize=(24, 10))
fig.suptitle('Latent Space Interpolation', fontsize=20)
plt.axis('off')


for i in range(11):
    minifig= fig.add_subplot(2, 6, i+1)
    image = netG(init_noisez1.cpu(), delta*i*init_noisez2.cpu()).detach().cpu().view(-1, 3, 128, 128)
    discr_result = netD(image).detach().numpy()
    class_result,_ = Classifier(image)
    class_result = class_result.numpy()
    minifig.axis('off')
    
    if (i==0):
        minifig.title.set_text("Negative:"+"\n"+"Class: "+str(class_result[0]))
    elif (i==10):
        minifig.title.set_text("Final Positive:"+"\n"+"Class: "+str(class_result[0]))
    else: 
        minifig.title.set_text("Class: "+str(class_result[0]))
    plt.imshow((np.transpose(image[0],(1,2,0))+1)/2)
  

# difference image
minifig= fig.add_subplot(2, 6, 12)
minifig.title.set_text('Positive-Negative Difference')
minifig.axis('off')
pos = netG(init_noisez1.cpu(), init_noisez2.cpu()).detach().cpu().view(-1, 3, 128, 128)
neg = netG(init_noisez1.cpu(), 0*init_noisez2.cpu()).detach().cpu().view(-1, 3, 128, 128)
diffimg = pos - neg
plt.imshow((np.transpose(diffimg[0],(1,2,0))+1)/2)

