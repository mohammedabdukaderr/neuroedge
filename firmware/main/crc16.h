/**
 * CRC16-CCITT (polynomial 0x1021, init 0xFFFF) — used for all UART frames.
 */
#ifndef NE_CRC16_H
#define NE_CRC16_H

#include <stdint.h>
#include <stddef.h>

/**
 * Compute CRC16-CCITT over a byte buffer.
 * @param data  Pointer to data bytes
 * @param len   Number of bytes
 * @return      16-bit CRC
 */
uint16_t ne_crc16(const uint8_t *data, size_t len);

/**
 * Format a CRC as a 4-character uppercase hex string (no 0x prefix).
 * buf must be at least 5 bytes.
 */
void ne_crc16_to_hex(uint16_t crc, char *buf);

/**
 * Parse a 4-char hex string to uint16_t.
 * Returns 0xFFFF on parse error (unlikely to be a valid CRC in practice).
 */
uint16_t ne_crc16_from_hex(const char *hex);

#endif /* NE_CRC16_H */
