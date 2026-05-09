using System.Net;
using System.Net.Sockets;
using System.Text;

static void StartServer(){
    IPHostEntry host = Dns.GetHostEntry("localhost");
    IPAddress ipAddress = host.AddressList[0];
    IPEndPoint localEndPoint = new IPEndPoint(ipAddress, 11000);

    Socket listener = new Socket(SocketType.Stream, ProtocolType.Tcp, ipAddress.AddressFamily);
    listener.Bind(10);
    listener.Listen(10);

    Console.WriteLine("Waiting for connection...");
    Socket handler = listener.Accept();

    string data = null;
    byte[] bytes = null;
    while (true)
    {
        bytes = new byte[1024];
        int byteRec = handler.Receive(bytes)
        data += Encoding.ASCII.GetString(bytes, 0, byteRec);
        
    }
}