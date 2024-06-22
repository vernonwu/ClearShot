import paramiko  # 用于调用scp命令
from scp import SCPClient
 
def connect_scp():
    host = "connect.cqa1.seetacloud.com"
    port = 17966
    username = "root"  # ssh 用户名
    password = "/PA6Bn82LFFY"  # 密码
 
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(host, port, username, password)
    scpclient = SCPClient(ssh_client.get_transport(),socket_timeout=1000.0)
    return scpclient
# 将指定目录的图片文件上传到服务器指定目录
def scp_upload_img(name):
    host = "connect.cqa1.seetacloud.com"  #服务器ip地址
    port = 17966  # 端口号
    username = "root"  # ssh 用户名
    password = "/PA6Bn82LFFY"  # 密码
 
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(host, port, username, password)
    scpclient = SCPClient(ssh_client.get_transport(),socket_timeout=15.0)

    scpclient.put("C:/Users/Administrator/aisys/fastapi-main/post_image/"+name , "projects/FFTformer/media/pred/input/1/")
    #scpclient.put("C:/Users/Administrator/aisys/fastapi-main/post_image/put.txt", "projects/FFTformer/media/pred/input/1/")
    ssh_client.close()

def scp_download_img(name):
    host = "connect.cqa1.seetacloud.com"  #服务器ip地址
    port = 17966  # 端口号
    username = "root"  # ssh 用户名
    password = "/PA6Bn82LFFY"  # 密码
 
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(host, port, username, password)
    scpclient = SCPClient(ssh_client.get_transport(),socket_timeout=15.0)

    scpclient.get("projects/FFTformer/results/Adaptive_fftformer/output/1/"+name,"C:/Users/Administrator/aisys/fastapi-main/get_image/")
if __name__ == "__main__":
    scp_download_img("2024-06-03-15-12-50.jpg")