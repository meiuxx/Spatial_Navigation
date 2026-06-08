using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class socketListener()
{
    public static int Main(String[] args)
    {
        StartServer();
        return 0;
    }

    public static void StartServer()
    {
        UdpClient udpc = new UdpClient(7878);
        IPEndPoint ep = null;

        while (true)
        {
            Console.WriteLine("Enter Student name":);
            String name = Console.ReadLine();
            byte[] msg = Encoding.ASCII.GetBytes(name);
            udpc.Send(msg, msg.Length);
            

        }
        
    }
}