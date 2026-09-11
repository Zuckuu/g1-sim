// Read-only RS-485 probe for BrainCo Revo 2 hands on the G1 Jetson: opens each /dev/ttyUSB*, asks the usual Modbus
// slave IDs for device info, never moves a finger. Found the left hand on ttyUSB2/slave 126 when the service had missed it.
// Build on the Jetson (needs brainco_hand_service checked out for stark-sdk.h + libbc_stark_sdk.so):
//   g++ -O2 -I$HOME/brainco_hand_service/include -L$HOME/brainco_hand_service/lib/aarch64 -o /tmp/probe_hands probe_hands.cpp \
//       -lbc_stark_sdk -lpthread -ldl -Wl,-rpath,$HOME/brainco_hand_service/lib/aarch64
//   sudo /tmp/probe_hands          # sudo: unitree is not in dialout; stop brainco_hand.service first to probe the bound ports
#include "stark-sdk.h"
#include <cstdio>
#include <string>
#include <vector>

static const char* sku_name(int s) {
    switch (s) {
        case 1: return "MEDIUM_RIGHT";
        case 2: return "MEDIUM_LEFT";
        case 3: return "SMALL_RIGHT";
        case 4: return "SMALL_LEFT";
        default: return "UNKNOWN";
    }
}

static void try_ids(const char* port, uint32_t baud, const std::vector<uint8_t>& ids) {
    DeviceHandler* h = modbus_open(port, baud);
    if (!h) {
        printf("  open FAIL %s @ %u\n", port, baud);
        return;
    }
    printf("  open OK   %s @ %u\n", port, baud);
    for (uint8_t id : ids) {
        CDeviceInfo* info = stark_get_device_info(h, id);
        if (!info) {
            printf("    slave %3u (0x%02x): no device info\n", id, id);
            continue;
        }
        printf("    slave %3u (0x%02x): sku=%s(%d) hw=%d fw=%s sn=%s\n",
               id, id, sku_name(info->sku_type), (int)info->sku_type,
               (int)info->hardware_type,
               info->firmware_version ? info->firmware_version : "?",
               info->serial_number ? info->serial_number : "?");
        free_device_info(info);
    }
    modbus_close(h);
}

int main() {
    init_logging(LogLevel::LOG_LEVEL_WARN);
    printf("=== list_available_ports ===\n");
    list_available_ports();

    const char* ports[] = {"/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3"};
    std::vector<uint8_t> ids = {1, 2, 10, 126, 127};

    for (const char* p : ports) {
        printf("\n=== brute %s ===\n", p);
        try_ids(p, 460800, ids);
        try_ids(p, 115200, ids);
        printf("  auto_detect_modbus_revo2(%s, quick=true)...\n", p);
        CDeviceConfig* cfg = auto_detect_modbus_revo2(p, true);
        if (!cfg) {
            printf("    none\n");
        } else {
            printf("    FOUND protocol=%d port=%s baud=%u slave=%u (0x%02x)\n",
                   (int)cfg->protocol, cfg->port_name ? cfg->port_name : "?",
                   cfg->baudrate, cfg->slave_id, cfg->slave_id);
            free_device_config(cfg);
        }
    }
    return 0;
}
