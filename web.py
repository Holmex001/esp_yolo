import paho.mqtt.client as mqtt
import json
import time

# ================= 配置区 =================
# 如果你还在用免费测试服，就填 "broker.emqx.io"，端口 1883
# 如果你的云服务器 Mosquitto 已经配好，就填你的域名 "mqtt.holmexsherlock.top"，端口 1883
MQTT_BROKER = "mqtt.holmexsherlock.top"
MQTT_PORT = 1883

TOPIC_SUBSCRIBE = "ESP32/To/App"  # 监听 ESP32 发来的原始数据
TOPIC_PUBLISH_CMD = "App/To/ESP32"  # 用来给 ESP32 下发指令
TOPIC_PUBLISH_APP = "Python/To/App"  # 🌟 新增：Python 专门给前端 App 发消息的主题


# ==========================================

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[本地大脑] 成功连接到 MQTT 服务器!")
        client.subscribe(TOPIC_SUBSCRIBE)
        print(f"[本地大脑] 已订阅 ESP32 数据主题: {TOPIC_SUBSCRIBE}")
    else:
        print(f"[本地大脑] 连接失败，返回码 {rc}")


def on_message(client, userdata, msg):
    payloadStr = msg.payload.decode("utf-8")

    try:
        # 1. 解析 ESP32 发来的 JSON 数据
        data = json.loads(payloadStr)
        lux_value = float(data.get("lux", 0))
        temp = data.get("temp", "未知")
        hum = data.get("hum", "未知")

        print(f"\n[收到数据] 温度: {temp}°C, 湿度: {hum}%, 光强: {lux_value} Lux")

        # 2. 核心业务逻辑：如果光线太暗，不仅让 ESP32 开灯，还要告诉前端 App
        if lux_value < 50.0:
            print("[智能控制] 触发暗光报警！")

            # 让 ESP32 开灯
            client.publish(TOPIC_PUBLISH_CMD, "ON")

            # 把报警信息打包成 JSON，发给前端 App
            alert_msg = json.dumps({
                "alert": True,
                "msg": "光线过暗，系统已自动开灯！",
                "time": time.strftime("%H:%M:%S")
            }, ensure_ascii=False)

            client.publish(TOPIC_PUBLISH_APP, alert_msg)
            print(f"--> 已向前端 App 推送报警信息: {alert_msg}")

    except Exception as e:
        print(f"[警告] JSON 解析失败，收到的非标准数据: {payloadStr}，错误: {e}")


def main():
    client = mqtt.Client("PythonLocalBackend_001")
    client.on_connect = on_connect
    client.on_message = on_message

    print("[本地大脑] 正在尝试连接服务器...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[本地大脑] 服务已手动停止")
        client.disconnect()
    except TimeoutError:
        print("\n[报错] 连接超时！请检查云服务器 1883 端口是否在安全组中放行。")


if __name__ == "__main__":
    main()