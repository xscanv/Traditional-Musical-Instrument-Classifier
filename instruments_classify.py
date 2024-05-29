import wx
import os

APP_TITLE = u'Traditional-Musical-Instrument-Classifier|传统乐器音色识别器'
APP_ICON = './Desktop/2024/2024A机器学习/Musical Instruments/main/favicon.ico' # 请更换成你的icon

def get_file_extension(file_path):
        # 获取文件的后缀名
        file_name, file_extension = os.path.splitext(file_path)
        return file_extension

class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title='My Frame')

        self.SetSize((800, 600))
        self.Center()
        # 创建菜单栏
        menubar = wx.MenuBar()
        # 创建文件菜单
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_OPEN, 'Open')
        menubar.Append(file_menu, '&File')
        # 设置菜单栏
        self.SetMenuBar(menubar)
        # 设置显示文字
        title = wx.StaticText(self, -1, 'Musical Instruments\nClassifier ver.1.0',  pos=(0, 10), size=(800, -1), style=wx.ALIGN_CENTRE_HORIZONTAL)
        title.SetForegroundColour('black') #设置字体颜色
        title.SetBackgroundColour('white') #设置背景颜色
        title.SetFont(wx.Font(48, wx.TELETYPE, wx.NORMAL, wx.BOLD))
        text = wx.StaticText(self, -1, 'This is a project demo for SJTU BUSS2505.\nThis program can help to sort out different Chinese musical instrument.\nTo start you can audio file via file buttom on top left.\nBy now we support wav only.\nGOOD LUCK!',  pos=(0, 150), size=(800, -1), style=wx.ALIGN_CENTRE_HORIZONTAL)
        text.SetForegroundColour('black') #设置字体颜色
        text.SetBackgroundColour('white') #设置背景颜色
        text.SetFont(wx.Font(14, wx.TELETYPE, wx.NORMAL, wx.LIGHT))
        self.textCtrl = wx.TextCtrl(self, -1,pos=(0,470),size=(800,-1),style=wx.TE_READONLY|wx.TE_AUTO_URL|wx.TE_RICH|wx.BORDER_NONE|wx.ALIGN_CENTRE_HORIZONTAL)
        attr = wx.TextAttr()
        attr.SetFontUnderlined(True)
        attr.SetTextColour(wx.BLUE)
        self.textCtrl.SetDefaultStyle(attr)
        self.url = 'https://github.com/xscanv/Traditional-Musical-Instrument-Classifier'
        text = 'Click this link for more info: {}'
        self.textCtrl.SetValue(text.format(self.url))
        self.Bind(wx.EVT_TEXT_URL, self.onLinkClicked, self.textCtrl)
        # 加载图像
        image = wx.Image('./Desktop/2024/2024A机器学习/Musical Instruments/fig/logo.png', wx.BITMAP_TYPE_ANY)
        # 获取图像控件的尺寸
        size = self.GetSize()
        # 调整图像的大小以适应控件
        image = image.Rescale(290, 210)
        # 创建StaticBitmap控件并显示图像
        bitmap = wx.StaticBitmap(self, wx.ID_ANY, wx.BitmapFromImage(image),pos=(250,250))
        self.Show(True)
        author = wx.StaticText(self, -1, 'SJTU XiaoMingyuan 2024.5.28',  pos=(0, 500), size=(800, -1), style=wx.ALIGN_CENTRE_HORIZONTAL)
        author.SetForegroundColour('grey') #设置字体颜色
        author.SetBackgroundColour('white') #设置背景颜色
        author.SetFont(wx.Font(18, wx.TELETYPE, wx.SLANT, wx.NORMAL))
        # 设置图标
        icon = wx.Icon(APP_ICON, wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)
        # 绑定事件处理函数
        self.Bind(wx.EVT_MENU, self.on_open, id=wx.ID_OPEN)
    def onLinkClicked(self, event):
        #这里做了一个鼠标左键按下判断（不判断会出现鼠标指向超链接就疯狂跳转）
        state = wx.GetMouseState()
        url='https://github.com/xscanv/Traditional-Musical-Instrument-Classifier'
        if state.LeftIsDown():
            #这里用的是webbrowser的open()
            #也可以用os的os.startfile(url)达到同样的效果；
            os.startfile(url)
    
    def get_file_extension(file_path):
        # 获取文件的后缀名
        file_name, file_extension = os.path.splitext(file_path)
        return file_extension
    
    def show_popup(self, event):
        # 创建一个弹出窗口
        popup = wx.PopupWindow(self, size=(200, 100))
        popup.SetPosition((150, 150))
        popup.SetBackgroundColour(wx.Colour(255, 255, 255))
        popup.SetForegroundColour(wx.Colour(0, 0, 0))

        # 在弹出窗口中添加文本
        text = wx.StaticText(popup, label='Unsupported file format, only WAV files are supported.', pos=(50, 30))

        # 显示弹出窗口
        popup.Show()

    def on_open(self, event):
        # 创建文件打开对话框
        dialog = wx.FileDialog(self, 'Open File', '', '', 'All Files (*.*)|*.*', wx.FD_OPEN)
        # 显示文件打开对话框
        if dialog.ShowModal() == wx.ID_OK:
            # 获取用户选择的文件路径
            filepath = dialog.GetPath()
            if(get_file_extension(filepath)=='.wav'):
                # 格式正确则读取文件内容
                instrument_pre, trust=m(filepath,modelpath)

            else:
                # 如文件格式不正确，弹出窗口并停止运行
                wx.FutureCall(1, self.ShowMessage)

    def ShowMessage(self):
        wx.MessageBox('Unsupport File!', 'Warning', wx.OK | wx.ICON_ERROR)

                

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()