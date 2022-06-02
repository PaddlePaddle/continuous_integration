import smtplib
from email.mime.text import MIMEText
from email.header    import Header

def send_mail(sender_addr, receiver_addr, subject, content):
    msg =  MIMEText(content, 'html', 'UTF-8')
    msg['From'] = sender_addr
    msg['To'] = receiver_addr
    msg['Subject'] = Header(subject, 'UTF-8')

    server = smtplib.SMTP()
    server.connect("proxy-in.baidu.com")
    try:
        server.sendmail(sender_addr, msg['To'].split(','), msg.as_string())
        print("email send")
    except Exception as e:
        print("发送邮件失败:%s" % (e))
    finally:
        server.quit()


if __name__ == "__main__":
    sender_addr = "paddle_benchmark@baidu.com"
    receiver_addr = "zhengya01@baidu.com"
    subject = "【TIPC:chain_base:PaddleOCR:20220602】运行结果"
    content = """
<html>
    <body>
        <div style="text-align:center;">
        <a align=center href="BENCHMARK_PIPELINE_ADDR">任务参考链接</a><br/><br/>
        </div>
        <HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption bgcolor="#989898">任务执行情况汇总</caption>
        <tr><td></td><td>成功</td><td>失败</td><td>超时</td></tr>
        <tr><td>模型</td><td>5</td><td>2</td><td>1</td></tr>
        <tr><td>case</td><td>50</td><td>3</td><td>-</td></tr>
        </table>
        <br><br>
        <table border="1" align=center>
        <caption bgcolor="#989898">失败列表</caption>
        <tr><td>模型</td><td>case</td></tr>
        <tr><td>m1</td><td>case1</td></tr>
        <tr><td>m1</td><td>case2</td></tr>
        <tr><td>m2</td><td>case1</td></tr>
        </table>
    </body>
</html>
"""
    send_mail(sender_addr, receiver_addr, subject, content)
